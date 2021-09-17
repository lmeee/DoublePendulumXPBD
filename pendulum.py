from posixpath import join
import torch
import pytorch3d.transforms as tf
from torch.functional import norm
import numpy as np
import matplotlib.pyplot as plt

maxRotationPerSubstep = torch.tensor(0.5)
PI = torch.tensor(3.14159265359)

'''
tf Quaternion multiply will automatically change the sign of rotation, which is not wanted 
should use Quaternion raw multiply which could preserve the sign of multiplicationj
'''


def Plot2dPendulumPosition(q1, q2, x1, x2):
    gndpt = np.array([0.,0.75])
    orij1 = np.array([0.,0.75])
    orij2 = np.array([2.,0.75])
    oript = np.concatenate((gndpt, orij1, orij2))
    rb1_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([0.,-1.,0.]))+x1)[1:]
    rb1_j2_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([0.,1.,0.]))+x1)[1:]
    rb1 = np.concatenate((rb1_j1_pt.numpy(), x1.numpy()[1:], rb1_j2_pt.numpy()))

    rb2_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q2), torch.tensor([0.,-1.,0.]))+x2)[1:]
    rb2_j2_pt = (torch.matmul(tf.quaternion_to_matrix(q2), torch.tensor([0.,1.,0.]))+x2)[1:]
    rb2 = np.concatenate((rb2_j1_pt.numpy(), x2.numpy()[1:], rb2_j2_pt.numpy()))
    yidx = [0,2,4]
    zidx = [1,3,5]

    # print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-rb1_j2_pt), np.linalg.norm(rb2_j1_pt-rb2_j2_pt))
    print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-orij1), np.linalg.norm(rb1_j2_pt-rb2_j1_pt))
    print('quaternion x', q1[1], q2[1])
    plt.plot(oript[yidx], oript[zidx], 'ro')
    plt.plot(rb1[yidx], rb1[zidx], 'bo')
    plt.plot(rb2[yidx], rb2[zidx], 'go')
    plt.arrow(rb1[0], rb1[1], rb1[4]-rb1[0], rb1[5]-rb1[1], width=0.03)
    plt.arrow(rb2[0], rb2[1], rb2[4]-rb2[0], rb2[5]-rb2[1], width=0.03)
    plt.axis('equal')

    plt.show()
    # plt.pause(0.01)
    plt.clf()

def Plot2dPendulumPositionFei(q1, q2, x1, x2, iter, title=''):
    gndpt = np.array([0.,0.])
    orij1 = np.array([0.4,0.])
    orij2 = np.array([0.8,0.])
    oript = np.concatenate((gndpt, orij1, orij2))
    rb1_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([-0.2,0.,0.]))+x1)[0:2]
    rb1_j2_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([0.2,0.,0.]))+x1)[0:2]
    rb1 = np.concatenate((rb1_j1_pt.numpy(), x1.numpy()[0:2], rb1_j2_pt.numpy()))

    rb2_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q2), torch.tensor([-0.2,0.,0.]))+x2)[0:2]
    rb2_j2_pt = (torch.matmul(tf.quaternion_to_matrix(q2), torch.tensor([0.2,0.,0.]))+x2)[0:2]
    rb2 = np.concatenate((rb2_j1_pt.numpy(), x2.numpy()[0:2], rb2_j2_pt.numpy()))
    yidx = [0,2,4]
    zidx = [1,3,5]

    # print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-rb1_j2_pt), np.linalg.norm(rb2_j1_pt-rb2_j2_pt))
    # print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-gndpt), np.linalg.norm(rb1_j2_pt-rb2_j1_pt))
    print('quaternion z', q1[3], q2[3])
    plt.plot(oript[yidx], oript[zidx], 'ro')
    plt.plot(rb1[yidx], rb1[zidx], 'bo')
    plt.plot(rb2[yidx], rb2[zidx], 'go')
    plt.arrow(rb1[0], rb1[1], rb1[4]-rb1[0], rb1[5]-rb1[1], width=0.03)
    plt.arrow(rb2[0], rb2[1], rb2[4]-rb2[0], rb2[5]-rb2[1], width=0.03)
    plt.axis('equal')
    plt.title(title)

    # plt.show()
    plt.pause(0.02)
    plt.savefig('{:02d}'.format(iter)+'dpen.png')
    plt.clf()

def PrintError(q1, q2, x1, x2):
    orij1 = np.array([0.,0.75])
    rb1_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([0.,-1.,0.]))+x1)[1:]
    rb1_j2_pt = (torch.matmul(tf.quaternion_to_matrix(q1), torch.tensor([0.,1.,0.]))+x1)[1:]

    rb2_j1_pt = (torch.matmul(tf.quaternion_to_matrix(q2), torch.tensor([0.,-1.,0.]))+x2)[1:]
    # print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-rb1_j2_pt), np.linalg.norm(rb2_j1_pt-rb2_j2_pt))
    print('error for constraint solve ', np.linalg.norm(rb1_j1_pt-orij1), np.linalg.norm(rb1_j2_pt-rb2_j1_pt))
    print('quaternion x', q1[1], q2[1])

class Pose:
    def __init__(self):
        self.p = torch.zeros(3)
        self.q = torch.tensor([1.,0.,0.,0.])


    def Clone(self):
        newpose = Pose()
        newpose.p = self.p.clone()
        newpose.q = self.q.clone()
        return newpose

    
    def Rotate(self,v):
        return tf.quaternion_apply(self.q, v)
    
    def invRotate(self, v):
        tmp = tf.quaternion_invert(self.q)
        return tf.quaternion_apply(tmp, v)
    
    def Transform(self, v):    
        return self.p + tf.quaternion_apply(self.q, v)
    
    def invTransform(self, v):
        v = v - self.p
        return self.invRotate(v)
    
    def TransformPose(self, pose):
        pose.q = tf.quaternion_raw_multiply(self.q, pose.q)
        pose.p = self.Rotate(pose.p)
        pose.p = pose.p + self.p
        return pose

def getQuatAxis0(q):
    x2 = q[1] * 2.0
    w2 = q[0] * 2.0
    return torch.tensor([(q[0] * w2) - 1.0 + q[1] * x2, (q[3] * w2) + q[2] * x2, (-q[2] * w2) + q[3] * x2])

def getQuatAxis1(q):
    y2 = q[2] * 2.0
    w2 = q[0] * 2.0
    return torch.tensor([(-q[3] * w2) + q[1] * y2, (q[0] * w2) - 1.0 + q[2] * y2, (q[1] * w2) + q[3] * y2])

def getQuatAxis2(q):
    z2 = q[3] * 2.0
    w2 = q[0] * 2.0
    return torch.tensor([(q[2] * w2) + q[1] * z2, (-q[1] * w2) + q[2] * z2, (q[0] * w2) - 1.0 + q[3] * z2])
 

# Rigid Body Class
class RigidBody:
    pose = Pose()
    prevPose = Pose()
    origPose = Pose()
    def __init__(self, pose):
        self.pose = pose.Clone()
        self.prevPose = pose.Clone()
        self.origPose = pose.Clone()
        self.vel = torch.tensor([0.0, 0.0, 0.0])
        self.omega = torch.tensor([0.0, 0.0, 0.0])
        self.invMass = torch.tensor(1.0)
        self.invInertia = torch.tensor([1.0, 1.0, 1.0])

    def SetBox(self, size, density = 1.0) :
        mass = size[0] * size[1] * size[2] * density
        self.invMass = 1.0 / mass
        mass = mass /12.0
        self.invInertia = torch.tensor([
            1.0 / (size[1] * size[1] + size[2] * size[2]) / mass,
            1.0 / (size[2] * size[2] + size[0] * size[0]) / mass,
            1.0 / (size[0] * size[0] + size[1] * size[1]) / mass])

    def ApplyRotation(self, rot, scale = 1.0):

        # // safety clamping. self happens very rarely if the solver
        # // wants to turn the body by more than 30 degrees in the
        # // orders of milliseconds

        maxPhi = 0.5
        phi = torch.norm(rot)
        if (phi * scale > maxRotationPerSubstep) :
            print('limit hit')
            scale = maxRotationPerSubstep / phi
            
        dq = torch.tensor([0., rot[0] * scale, rot[1] * scale, rot[2] * scale])					
        dq = tf.quaternion_raw_multiply(dq, self.pose.q)
        self.pose.q = torch.tensor([ self.pose.q[0] + 0.5 * dq[0], 
                self.pose.q[1] + 0.5 * dq[1], self.pose.q[2] + 0.5 * dq[2], 
                self.pose.q[3] + 0.5 * dq[3]])
        self.pose.q = self.pose.q / torch.norm(self.pose.q)


    def integrate(self, dt, gravity, torque):
        self.prevPose = self.pose.Clone()
        self.vel = self.vel + gravity*dt					
        self.pose.p = self.pose.p + self.vel * dt

        # apply torque
        torque = torch.tensor([0., 0., torque])
        self.omega = self.omega + (self.invInertia[2]) * torque * dt

        self.ApplyRotation(self.omega, dt)


    def update(self, dt):
        self.vel = self.pose.p - self.prevPose.p
        self.vel = self.vel / dt
        dq = tf.quaternion_raw_multiply(self.pose.q, tf.quaternion_invert(self.prevPose.q))
        self.omega = torch.tensor([dq[1], dq[2], dq[3]]) * 2.0 / dt
        if (dq[0] < 0.0):
            self.omega = -self.omega

        # // self.omega.multiplyScalar(1.0 - 1.0 * dt)
        # // self.vel.multiplyScalar(1.0 - 1.0 * dt)

    # why no dt??
    def getVelocityAt(self, pos):					
        vel = pos - self.pose.p
        vel = torch.cross(vel, self.omega)
        vel = self.vel - vel
        return vel


    def getInverseMass(self, normal, pos = None):
        if pos is None :
            n = normal.clone()
        else:
            n = pos - self.pose.p
            n = torch.cross(n, normal)
    
        n = self.pose.invRotate(n)
        w = n[0] * n[0] * self.invInertia[0] +\
            n[1] * n[1] * self.invInertia[1] +\
            n[2] * n[2] * self.invInertia[2]
        if (pos is not None):
            w += self.invMass
        return w


    def applyCorrection(self, corr, pos = None, velocityLevel = False):
        if (pos is None) :
            dq = corr.clone()
        else:
            if velocityLevel:
                self.vel = self.vel + corr * self.invMass
            else:
                self.pose.p = self.pose.p + corr * self.invMass
            dq = pos - self.pose.p
            dq = torch.cross(dq, corr)
    
        dq = self.pose.invRotate(dq)
        dq = torch.tensor([self.invInertia[0] * dq[0], 
            self.invInertia[1] * dq[1], self.invInertia[2] * dq[2]])
        dq = self.pose.Rotate(dq)
        if velocityLevel:
            self.omega = self.omega + dq
        else :
            self.ApplyRotation(dq)

########
def ApplyBodyPairCorrection(body0, body1, corr, compliance, dt,
    pos0 = None, pos1 = None, velocityLevel = False) :
    C = torch.norm(corr)
    if (C == 0.0):
        return body0, body1

    normal = corr.clone()
    normal = normal / C

    if body0:
        w0 = body0.getInverseMass(normal, pos0)
    else:
        w0 = 0.
    if body1:
        w1 = body1.getInverseMass(normal, pos1)
    else:
        w1 = 0.
    w = w0 + w1
    if (w == 0.0):
        return body0, body1

    lam = -C / (w + compliance / dt / dt)
    normal = normal * -lam
    if body0:
        body0.applyCorrection(normal, pos0, velocityLevel)
    if body1:
        normal = normal * -1.0
        body1.applyCorrection(normal, pos1, velocityLevel)
    
    return body0, body1

def limitAngle(body0, body1, n, a, b, minAngle, maxAngle,
             compliance, dt, maxCorr = PI):

    # // the key function to handle all angular joint limits
    c = torch.cross(a, b)

    phi = torch.asin(torch.sum(torch.mul(c,n)))
    if (torch.sum(torch.mul(a,b)) < 0.0):
        phi = PI - phi

    if (phi > PI):
        phi -= 2.0 * PI
    if (phi < -PI):
        phi += 2.0 * PI

    if phi < minAngle or phi > maxAngle:
        phi = torch.min(torch.max(minAngle, phi), maxAngle)

        q = tf.axis_angle_to_quaternion(n * phi)

        omega = a.clone()
        omega = tf.quaternion_apply(q, omega)
        omega = torch.cross(omega, b)

        phi = torch.norm(omega)
        if (phi > maxCorr) :
            omega = omega * maxCorr / phi

        body0, body1 = ApplyBodyPairCorrection(body0, body1, omega, compliance, dt)

class Joint:
    # body0 = RigidBody()
    # body1 = RigidBody()
    localPose0 = Pose()
    localPose1 = Pose()
    globalPose0 = Pose()
    globalPose1 = Pose()
    def __init__(self, type, body0, body1, localPose0, localPose1):
        self.body0 = body0
        self.body1 = body1
        self.localPose0 = localPose0.Clone()
        self.localPose1 = localPose1.Clone()
        self.globalPose0 = localPose0.Clone()
        self.globalPose1 = localPose1.Clone()

        self.type = type					
        self.compliance = 0.0
        self.rotDamping = 0.0
        self.posDamping = 0.0
        self.hasSwingLimits = False
        self.minSwingAngle = -2.0 * PI
        self.maxSwingAngle = 2.0 * PI
        self.swingLimitsCompliance = 0.0
        self.hasTwistLimits = False
        self.minTwistAngle = -2.0 * PI
        self.maxTwistAngle = 2.0 * PI
        self.twistLimitCompliance = 0.0
        
    def updateGlobalPoses(self) :
        self.globalPose0 = self.localPose0.Clone()
        if (self.body0):
            self.globalPose0 = self.body0.pose.TransformPose(self.globalPose0)
        self.globalPose1 = self.localPose1.Clone()
        if (self.body1):
            self.globalPose1 = self.body1.pose.TransformPose(self.globalPose1)

    def solvePos(self, dt):

        self.updateGlobalPoses()

        if (self.type == 'hinge'):
            # // align axes
            a0 = getQuatAxis0(self.globalPose0.q)
            b0 = getQuatAxis1(self.globalPose0.q)
            # c0 = getQuatAxis2(self.globalPose0.q)
            a1 = getQuatAxis0(self.globalPose1.q)
            a0 = torch.cross(a0, a1)
            self.body0, self.body1 = ApplyBodyPairCorrection(
                                self.body0, self.body1, a0, 0.0, dt)
            # // limits
            if (self.hasSwingLimits):
                self.updateGlobalPoses()
                n = getQuatAxis0(self.globalPose0.q)
                b0 = getQuatAxis1(self.globalPose0.q)
                b1 = getQuatAxis1(self.globalPose1.q)
                limitAngle(self.body0, self.body1, n, b0, b1, 
                    self.minSwingAngle, self.maxSwingAngle, self.swingLimitsCompliance, dt)

        # // simple attachment
        self.updateGlobalPoses()
        corr = self.globalPose1.p - self.globalPose0.p
        self.body0, self.body1 = ApplyBodyPairCorrection(self.body0, self.body1, 
            corr, self.compliance, dt,
            self.globalPose0.p, self.globalPose1.p)	

    def solveVel(self, dt) : 

        # // Gauss-Seidel lets us make damping unconditionally stable in a 
        # // very simple way. We clamp the correction for each constraint
        # // to the magnitude of the currect velocity making sure that
        # // we never subtract more than there actually is.

        if (self.rotDamping > 0.0) :
            omega = torch.tensor([0.0, 0.0, 0.0])
            if (self.body0):
                omega = omega - self.body0.omega
            if (self.body1):
                omega = omega + self.body1.omega
            omega = omega * torch.min(1.00, self.rotDamping * dt)
            self.body0, self.body1 = ApplyBodyPairCorrection(self.body0, self.body1, 
                    omega, 0.0, dt, 
                    None, None, True)
        if (self.posDamping > 0.0) :
            self.updateGlobalPoses()
            vel = torch.tensor([0.0, 0.0, 0.0])
            if (self.body0):
                vel = vel - self.body0.getVelocityAt(self.globalPose0.p)
            if (self.body1):
                vel = vel + self.body1.getVelocityAt(self.globalPose1.p)
            vel = vel * torch.min(1.0, self.posDamping * dt)
            self.body0, self.body1 = ApplyBodyPairCorrection(self.body0, self.body1, vel, 0.0, dt, 
                    self.globalPose0.p, self.globalPose1.p, True)

def DoublePendulum():
    bodies = []
    joints = []

    # body initialization
    for i in range(2):
        pos = Pose()
        pos.p = torch.zeros(3)
        pos.p[2] = 0.75
        pos.p[1] = 1. + i*2.
        pos.q = torch.tensor([1.,0.,0.,0.])
        rb = RigidBody(pos)
        rb.SetBox(torch.tensor([0.4,2.,0.4]), 1./0.32)
        bodies.append(rb)

    localpose0 = Pose()
    localpose0.p = torch.tensor([0,0.,0.75])
    localpose0.q = torch.tensor([1.,0.,0.,0.])
    localpose1 = Pose()
    localpose1.p = torch.tensor([0.,-1.,0.])
    localpose1.q = torch.tensor([1.,0.,0.,0.])
    localpose2 = Pose()
    localpose2.p = torch.tensor([0.,1.,0.])
    localpose2.q = torch.tensor([1.,0.,0.,0.])
    joints.append(Joint('hinge', None, bodies[0], 
                localpose0, localpose1))

    joints.append(Joint('hinge', bodies[0], bodies[1], 
                localpose2, localpose1.Clone()))
    Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q,
                            bodies[0].pose.p, bodies[1].pose.p)
    timesteps = 0.05
    numSubsteps = 150
    dt = timesteps / numSubsteps
    gravity = torch.tensor([0.,0.,-10.])

    gracompen_torque = torch.tensor([40.,10.])

    for iter in range(int(3./timesteps)):
        print('current time frame ', iter)
        for i in range(numSubsteps):
            for j in range(len(bodies)):
                bodies[j].integrate(dt, gravity, gracompen_torque[j])
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # PrintError(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            for j in range(len(joints)):
                joints[j].solvePos(dt)
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # PrintError(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            for j in range(len(bodies)):
                bodies[j].update(dt)
            # PrintError(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # for j in range(len(joints)):
            #     joints[j].solveVel(dt)
        Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)

def DoublePendulumFei(gracompen_torque, title):
    bodies = []
    joints = []
    numSubsteps = 40
    timesteps = 1/60.
    dt = timesteps /numSubsteps
    gravity = torch.tensor([0.,-10., 0.])

    # body initialization 1
    bodysize = torch.tensor([0.4, 0.05, 0.05])
    pose = Pose()
    pose.p = torch.tensor([0.2, 0.,0.])
    boxBody = RigidBody(pose)
    boxBody.SetBox(bodysize)
    bodies.append(boxBody)

    # body initialization 2
    pose = Pose()
    pose.p = torch.tensor([0.6, 0.,0.])
    boxBody = RigidBody(pose)
    boxBody.SetBox(bodysize)
    bodies.append(boxBody)

    jointPose0 = Pose()
    jointPose1 = Pose()
    jointPose0.q = tf.axis_angle_to_quaternion(torch.tensor([0., -0.5*PI, 0.]))
    jointPose1.q = tf.axis_angle_to_quaternion(torch.tensor([0., -0.5*PI, 0.]))

    jointPose0.p = torch.tensor([-0.2, 0., 0.])
    jointPose1.p = torch.tensor([0.2, 0., 0.])

    jointPose1 = jointPose0.Clone()
    jointPose1.p = bodies[0].pose.p + jointPose1.p

    joint1 = Joint('hinge', bodies[0], None, jointPose0, jointPose1)
    joints.append(joint1)

    jointPose0 = Pose()
    jointPose1 = Pose()
    jointPose0.q = tf.axis_angle_to_quaternion(torch.tensor([0., -0.5*PI, 0.]))
    jointPose1.q = tf.axis_angle_to_quaternion(torch.tensor([0., -0.5*PI, 0.]))
    
    jointPose0.p = torch.tensor([-0.2, 0., 0.])
    jointPose1.p = torch.tensor([0.2, 0., 0.])

    joint2 = Joint('hinge', bodies[1], bodies[0], jointPose0, jointPose1)
    joints.append(joint2)

    for iter in range(int(3./timesteps)):
        print('current time frame ', iter)
        for i in range(numSubsteps):
            for j in range(len(bodies)):
                bodies[j].integrate(dt, gravity, gracompen_torque[j])
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # print(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            for j in range(len(joints)):
                joints[j].solvePos(dt)
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # print(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            for j in range(len(bodies)):
                bodies[j].update(dt)
            # print(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # Plot2dPendulumPosition(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p)
            # for j in range(len(joints)):
            #     joints[j].solveVel(dt)
        Plot2dPendulumPositionFei(bodies[0].pose.q, bodies[1].pose.q, bodies[0].pose.p, bodies[1].pose.p, iter, title)


if __name__ == "__main__":

    ### Gravity Compensation stable point in simulation
    gracompen_torque = torch.tensor([6*10**-3.,2*10**-3])
    DoublePendulumFei(gracompen_torque, 'Gravity Compensation Stable Point in Sim')

    ### Gravity Compensation point calculated by Euler-Lagrange method
    # gracompen_torque = torch.tensor([8*10**-3.,2*10**-3])
    # DoublePendulumFei(gracompen_torque, 'Gravity Compensation Euler-Lagrange Result')