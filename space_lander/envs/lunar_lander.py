# from gym.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H, SCALE, LunarLander, EzPickle, Box2D, spaces, ContactDetector, edgeShape

import Box2D
import gym
import numpy as np
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)
from gym import spaces
from gym.envs.box2d import LunarLander
from gym.envs.box2d.lunar_lander import ContactDetector
from gym.utils import EzPickle

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

VIEWPORT_W = 1200
VIEWPORT_H = 600

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [
    (-10, -10), (-10, 50), (10, 50), (0, 60), (10, -10)
]

PLATFORM_WIDTH = 120
PLATFORM_HEIGHT = 10
PLATFORM_POLY = [
    (-PLATFORM_WIDTH / 2, PLATFORM_HEIGHT / 2),
    (PLATFORM_WIDTH / 2, PLATFORM_HEIGHT / 2),
    (PLATFORM_WIDTH / 2, -PLATFORM_HEIGHT / 2),
    (-PLATFORM_WIDTH / 2, -PLATFORM_HEIGHT / 2)]

LEG_AWAY = 12
LEG_DOWN = 18
LEG_W, LEG_H = 3, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 32.0
SIDE_ENGINE_AWAY = 12.0


class LunarLanderv1(LunarLander):
    
    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        
        self.world = Box2D.b2World(Box2D.b2Vec2(0, -1.6))
        self.moon = Box2D.b2World(Box2D.b2Vec2(0, -0.016))
        self.lander = None
        self.particles = []
        
        self.prev_reward = None
        
        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,),
                                            dtype=np.float32)
        
        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        
        # terrain
        CHUNKS = 22
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i
                    in range(CHUNKS)]
        
        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)
        
        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)
        
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(
                    VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        
        self.drawlist = [self.lander] + self.legs
        
        return self.step(np.array([0, 0]) if self.continuous else 0)[0]


gym.envs.register(
    id='LunarLanderv2-v0',
    entry_point='space_lander.envs.lunar_lander:LunarLanderv1',
    max_episode_steps=1000,
)
