import math # see xkcd.com/435

#
# n-body physics simulation
# copyright 2022 ryland goldman, all rights reserved
# advanced science research, los gatos high school
#
#
# changelog:
# version 0.0.0 - 11 sept 2022
#   it begins
#   defined the basic structure of this program
#   cpu only, also lossless calculations that will take forever
#

# constants, 1 now because units don't exist when there's nothing to compare them to
G = 1 # gravitational constant
k = 1 # coulumb constant

# particles
particles = [] # this is an empty universe rn (FIX)

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

# calculates interactions between particles
def particle_interaction(p1,p2):
    r = math.sqrt( (p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)    # 3d distance formula
    force = (G*p1.m*p2.m/(r**2)) + (-k*p1.q*p2.q/(r**2))                # graviational force (Gmm/r^2) plus electromagnetic force (kqq/r^2)
    acc_p1, acc_p2 = force/p1.m, force/p2.m                             # convert to acceleration via newton's 2nd law
    # --- yay trigonometry time ---
    # need to convert acc_p1 and acc_p2 to component to add them together (FIX)
    # i'm not doing this now because i'm lazy

n = 0 # counter
# main loop of program
def main():
    global particles, G, k, n
    while True: # loop forever i guess, maybe change later (FIX)
        n = n+1 # increase counter
        for p1 in particles:
            for p2 in particles: # nested for loops suck, find a better way (FIX)
                particle_interaction(p1, p2)
        input("frame "+str(n)+" completed, press enter to continue")
