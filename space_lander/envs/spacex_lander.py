import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.lunar_lander import ContactDetector, LunarLander
from gym.utils import seeding, EzPickle

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 15.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 26.0
SIDE_ENGINE_POWER  =  2.0

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY = [
	(-10, -10), (-10, 50), (10, 50), (0, 60), (10, -10)
]

PLATFORM_WIDTH = 120
PLATFORM_HEIGHT = 10
PLATFORM_POLY = [
    (-PLATFORM_WIDTH/2, PLATFORM_HEIGHT/2),
    (PLATFORM_WIDTH/2, PLATFORM_HEIGHT/2),
    (PLATFORM_WIDTH/2, -PLATFORM_HEIGHT/2),
    (-PLATFORM_WIDTH/2, -PLATFORM_HEIGHT/2)]

LEG_AWAY = 12
LEG_DOWN = 18
LEG_W, LEG_H = 3, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 32.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 1200
VIEWPORT_H = 600


class SpaceXLander(LunarLander):

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(Box2D.b2Vec2(0, -9.8))
        self.moon = None
        self.platform = None
        self.lander = None
        self.particles = []

        self.W = VIEWPORT_W/SCALE
        self.H = VIEWPORT_H/SCALE
        self.water_level = self.H * 0.1

        self.helipad_x1 = (self.W - (PLATFORM_WIDTH - 4)/SCALE)/2
        self.helipad_x2 = (self.W + (PLATFORM_WIDTH - 4)/SCALE)/2

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.platform)
        self.platform = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        self.platform = self.world.CreateStaticBody(
		position = (VIEWPORT_W/SCALE/2, self.water_level),
		angle=0.0,
		fixtures = fixtureDef(
			shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x, y in PLATFORM_POLY]),
			density=5.0,
		        restitution=0.0) # 0.99 bouncy
                )
        self.platform.color1 = (0.5,0.4,0.9)
        self.platform.color2 = (0.3,0.3,0.5)

        self.sky_polys = []
        self.water_polys = []
        p1 = (0, self.water_level)
        p2 = (self.W, self.water_level)

        self.sky_polys.append( [p1, p2, (p2[0],self.H), (p1[0],self.H)] )
        self.water_polys.append( [p1, p2, (p2[0], 0), (p1[0], 0)] )

        initial_y = self.H
        self.lander = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )

        self.lander.color1 = (0.5,0.5,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.03),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0,0]) if self.continuous else 0)[0]

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        #Wind
        self.lander.ApplyForceToCenter((-50, 0), True)

        #Air resistance
        p = 1.225 #kg/m^3
        A = 1 #cross sectional area
        Cd = 0.05 #drag coeff
        v_squared = (self.lander.linearVelocity.x ** 2 + self.lander.linearVelocity.y **2)
        drag_force = 0.5 * p * A * Cd * v_squared

        norm = self.lander.linearVelocity.length
        vel_x = self.lander.linearVelocity.x
        vel_y = self.lander.linearVelocity.y

        vel_x = vel_x * -drag_force / norm
        vel_y = vel_y * -drag_force / norm

        self.lander.ApplyForceToCenter((vel_x, vel_y), True)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.water_level+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state)==8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done   = True
            reward = -100
        if not self.lander.awake:
            done   = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0.6,0.8,0.85))
        for p in self.water_polys:
            self.viewer.draw_polygon(p, color=(0.05, 0.2, 0.4))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        trans = self.platform.fixtures[0].body.transform
        path = [trans* v for v in self.platform.fixtures[0].shape.vertices]
        self.viewer.draw_polygon(path, color=(1,1,1))
        path.append(path[0])
        self.viewer.draw_polyline(path, color=(0,0,0), linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.water_level
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

gym.envs.register(
    id='SpaceXLander-v0',
    entry_point='space_lander.envs.spacex_lander:SpaceXLander',
    max_episode_steps=1000,
)

