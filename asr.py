import math # physics is applied mathematics - xkcd.com/435
import time # sleep
import matplotlib.pyplot as plt # plotting results
import os # for ffmpeg
#
# n-body physics simulation
# copyright 2022 ryland goldman, all rights reserved
# advanced science research, los gatos high school
#
#
# changelog:
# version 0.0.0 - 13 sept 2022
#   it begins
#   defined the basic structure of this program
#   cpu only, also lossless calculations that will take forever
#
# version 0.1.0 - 23 sept 2022
#   added some particles
#   trig is in but doesnt work
#   probably missing a negative or a pi somewhere
#
# version 0.2.0 - 26 sept 2022
#   trig works (it was the arcsin function's fault)
#   to do: add collisions
#
# version 0.3.0 - 27 oct 2022
#   added time constant, decreased E
#   3d plotting enabled (matplotlib) for end result
#
# version 0.3.1 - 31 oct 2022
#   squished some bugs, it works now (more details in notebook)
#
# version 0.3.2 - 2 nov 2022
#   changed 3d plotting
#
 
# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 6.7e-11  # gravitational constant
k = 1 # coulumb constant
E = 1e-100 # softening constant
t = 1e-7 # time constant
 
# particles
particles = []
 
# structure of a particle has position, velocity, mass, and charge
class Particle:
    def __init__(self,x,y,z,vx,vy,vz,m,q):
        self.x = x # x-coordinate
        self.y = y # y-coordinate
        self.z = z # z-coordinate
        self.vx = vx # velocity vector (x-component)
        self.vy = vy # velocity vector (y-component)
        self.vz = vz # velocity vector (z-component)
        self.m = m # mass
        self.q = q # charge
    def __str__(self):
        print("Particle at (",str(self.x),",",str(self.y),",",str(self.z),") at |v|=(",str(self.vx),",",str(self.vy),",",str(self.vz),") - m=",str(self.m),", q=",str(self.q))
 
# calculates interactions between particles, define as a cuda kernel later for testing
def particle_interaction(p1,p2):
    if p1.x == p2.x and p1.y == p2.y and p1.z == p2.z:
        return p1 # don't have two of the same particles interact
    r = math.sqrt( (p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)    # 3d distance formula - sqrt( [x1-x2]^2 + [y1-y2] ^2 + [z1-z2]^2 )
    force = (G*p1.m*p2.m/((r+E)**2)) + (-k*p1.q*p2.q/((r+E)**2))        # gravitational force (Gmm/r^2) plus electromagnetic force (kqq/r^2)
    acc_p1 = - force/p1.m                                                 # convert to acceleration via newton's 2nd law
   
    # differences (NOT change/derivative)
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
   
    # calculate angles
    alpha = math.asin(dy/r)
    if dz == 0:
        beta = math.pi
    else:
         beta = math.atan(dx/(dz+0.001))

    if dx < 0: alpha = -alpha - math.pi
 
    # convert to component vectors, multiply by time constant, add
    p1.vx = p1.vx + acc_p1 * math.cos(alpha) * math.sin(beta) * t
    p1.vy = p1.vy + acc_p1 * math.sin(alpha) * t
    p1.vz = p1.vz + acc_p1 * math.cos(alpha) * math.cos(beta) * t
   
    return p1
 
def populate_universe():
    # again, units don't exist
    '''
    particles.append(Particle(1,5,9,1*t,-2*t,-1*t,1e10,0))
    particles.append(Particle(2,6,2,2*t,1*t,0,1e3,0))
    particles.append(Particle(3,4,1,-1*t,-1*t,2*t,1e5,0))
    particles.append(Particle(4,7,3,-2*t,2*t,2*t,1e8,0))
    particles.append(Particle(5,3,8,1*t,2*t,-1*t,1e7,0))
    particles.append(Particle(6,8,7,2*t,0,-1*t,1e8,0))
    particles.append(Particle(7,2,4,-1*t,1*t,2*t,1e2,0))
    particles.append(Particle(8,9,6,-2*t,1*t,0,1e9,0))
    particles.append(Particle(9,1,5,1*t,2*t,0,1e10,0))'''
    particles.append(Particle(0,0,0,0,0,0,2e30,0))
    particles.append(Particle(1.5e11,0,0,0,0,1e7*t,6e24,0))
 
n = 0 # counter n
# main loop of program
def main():
    global particles, G, k, n
    while n<50000: # outside loop
        if n%100 == 0:
            plot_particles(int(n/100))
        n = n+1 # increase counter
        for p2 in particles:
            #kick
            for p1 in particles:
                p1 = particle_interaction(p1, p2)
        for p in particles:
            #drift
            p.x = p.x + p.vx
            p.y = p.y + p.vy
            p.z = p.z + p.vz


# plot particles on graph
def plot_particles(itr_num):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    allX = []
    allY = []
    allZ = []
    for p in particles:
        allX.append(p.x);
        allY.append(p.y);
        allZ.append(p.z);
    ax.clear()
    ax.scatter3D(allX, allY, allZ)
    plt.savefig('/Users/rylandgoldman/Desktop/NbodySim/frame-'+str(itr_num)+'.png')
    plt.close()

populate_universe()
main()
plot_particles("end")

print("Simulation completed. Starting postprocessing...")
os.system("ffmpeg -f image2 -r 30 -i /Users/rylandgoldman/Desktop/NbodySim/frame-%01d.png -vcodec mpeg4 -y /Users/rylandgoldman/Desktop/NbodySim/video.mp4")#
# os.system("open /Users/rylandgoldman/Desktop/NbodySim/video.mp4")
print("Postprocessing completed. Check your desktop for the animation.")
