import numpy as np
from gurobipy import *
import numpy as np
import numpy as np
import bresenham as bham
import math
import time
import copy
import numpy as np
import cv2
import matplotlib as plt
import pylab as pl
from tifffile import imread, imsave
import cPickle as pickle
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def compute_weight_at( image, coords ):
    '''
    coords  list of lists containing as many entries as img has dimensions
    '''
    m = 0
    for c in coords:
        try:
            
            m = max( m, image[ tuple(c[::-1]) ] )
            x, y = tuple(c[::-1])
            
            ##Uncomment if you want eight max out of neighbors 
            #m = max( m, image[x-1, y-1] )
            #m = max( m, image[x-1, y+1] )
            #m = max( m, image[x+1, y+1] )
            #m = max( m, image[x+1, y-1] )
            #Max out of four neigbors and itself
            m = max( m, image[x-1, y] )
            m = max( m, image[x+1, y] )
            m = max( m, image[x, y-1] )
            m = max( m, image[x, y+1] )
   
        except:
            None
    return m

def sample_circle( n=18 ):
    '''
        Returns n many points on the unit circle (equally spaced).
    '''
    points = np.zeros([n,2])
    for i in range(n):
        angle = 2*math.pi * i/float(n)
        x = math.cos(angle)
        y = math.sin(angle)
        points[i] = [x,y]
    return points

def compute_vertex_cost(T, N, K, images, center, min_radius, max_radius):
    '''
        Returns weight matrix for all vertices per frame
        Cv = Delta(I) where I:intensity at v
    '''
    col_vectors = sample_circle( N )
    w = np.zeros([T, N, K])
    
    for t in range(T):
        for i in range(N):
            from_x = int(center[0] + col_vectors[i,0]*min_radius[0])
            from_y = int(center[1] + col_vectors[i,1]*min_radius[1])
            to_x = int(center[0] + col_vectors[i,0]*max_radius[0])
            to_y = int(center[1] + col_vectors[i,1]*max_radius[1])
            coords = bham.bresenhamline(np.array([[from_x, from_y]]), np.array([[to_x, to_y]]))
            
            num_pixels = len(coords)
            
            for k in range(K):
                start = int(k * float(num_pixels)/K)
                end = max( start+1, start + num_pixels/K )
                w[t, i, k] = -1 * compute_weight_at(images[t], coords[start:end])
                    
    return w

def compute_edge_cost(T, N, K, max_delta_k, alpha, beta, w):
    '''
        Returns weight matrix for all edges per frame
        Ce = alpha*Delta(I) + beta*Delta(X) 
        where,
        I:intensity at end nodes of 'e',  
        alpha, beta: smoothing parameters
    '''
    w_edge = np.zeros([T, N* K, N*K])
    INF = 9999999999
    w_edge.fill(INF)
    
    for t in range(T):
        for i in range( N ):
            for k in range(K):
                j=(i+1)%N 
                for l in range(K):
                    if (abs(k-l) <= max_delta_k):
                        c1 = alpha *(w[t][i][k] - w[t][j][l]) *(w[t][i][k] - w[t][j][l]) 
                        c2 = beta* (k-l)*(k-l)
                        w_edge[t][i*K + k][j*K +l] = c1 + c2 
                        test = test +1
    return w_edge


def compute_temporal_edge_cost(T, N, K, max_delta_time, alpha_t, beta_t, w):
    '''
        Returns weight matrix for eges between frames
        Ce_t = alpha_t *Delta(I) + beta_t *Delta(X) 
        
        I:intensity at end nodes of 'e',  
        alpha_t, beta_t: smoothing parameters
    '''
    w_e = np.ones((T, N*K, 2*K)) # time * me * my neighboring columns
    INF = 9999999999
    w_e.fill(INF)

    #across time
    for t in range(T):
        for i in range(N):
            for k in range(K):
                if t > 0 :
                    #left neighbor (time ) connections
                    for k_prime in range(K):
                        if (abs(k- k_prime) <= max_delta_time):
                            c1 = alpha_t * (w[t][i][k] - w[t-1][i][k_prime])*(w[t][i][k] - w[t-1][i][k_prime])
                            c2 = beta_t*(k- k_prime)*(k- k_prime)
                            w_e[t][i*K + k][k_prime] = c1 + c2 
                if t < T-1 :
                    #right neighbor (time) connection
                    for k_prime in range(K, K+K):
                        if (abs(k- (k_prime - K)) <= max_delta_time):
                            c1 = alpha_t * (w[t][i][k] - w[t+1][i][k_prime - K])*(w[t][i][k] - w[t+1][i][k_prime - K])
                            c2 = beta_t*(k- (k_prime - K))*(k- (k_prime - K))
                            w_e[t][i*K + k][k_prime] = c1 + c2
    return w_e



def get_surfaces(T, N, K, min_radius, max_radius, center, m, vv ):
    surfaces = []
    temp_vv = m.getAttr("X", vv)
    col_vectors = sample_circle(N)

    for t in range(T):
        surface = []
        for i in range(N):
            from_x = int(center[0] + col_vectors[i,0]*min_radius[0])
            from_y = int(center[1] + col_vectors[i,1]*min_radius[1])
            to_x = int(center[0] + col_vectors[i,0]*max_radius[0])
            to_y = int(center[1] + col_vectors[i,1]*max_radius[1])
            coords = bham.bresenhamline(np.array([[from_x, from_y]]), np.array([[to_x, to_y]]))
            num_pixels = len(coords)
            for k in range(K):
                if(temp_vv[(i*K+k)*T+t] == 1 and k !=0):
                    #print k
                    start = int(k * float(num_pixels)/K)
                    end = max( start+1, start + num_pixels/K )
                    k +=1
                    x = int(center[0] + col_vectors[i,0] * 
                    min_radius[0] + col_vectors[i,0] * 
                    (k-1)/float(K) * (max_radius[0]-min_radius[0]) )
                    y = int(center[1] + col_vectors[i,1] * 
                    min_radius[1] + col_vectors[i,1] * 
                    (k-1)/float(K) * (max_radius[1]-min_radius[1]) )
                    surface.append((x,y))
                    #surface.append((coords[(start)][0],coords[(start)][1]))
        surface.append(surface[0])
        surfaces.append(surface)
    return surfaces



def build_model(T, N, K, w, w_edge, w_e):
    #set model
    m = Model("model_1")
    INF = 9999999999
    '''
        Tasks: define variables, assign costs, define constraints, build model
    '''
    vv = {} #placeholder for vertices
    ve = {} #placeholder for edges
    cv = {} #placeholder for vertex cost
    ce = {} #placeholder for edge cost
    
    constr =0 #number of constraints
    
    ve_index = -1 * np.ones((T, 3, N,K,N,K)) #index set for ve and ce 
      
    cnt = 0
    expr = LinExpr() #expression for objective funtion 
    
    for t in range(T):
        for i in  range(N):
            expr2 = LinExpr() #expression for edge constraint
            for k in range(K):
                vv[(i*K+k)*T +t] = (m.addVar(vtype=GRB.BINARY, name="v " + str(i)+","+str(k) + ","+str(t))) 
                cv[(i*K+k)*T +t] = w[t][i][k]
                expr.add(vv[(i*K+k)*T +t], w[t][i][k])
                j = (i+1)%N
                for l in range(K):
                    if(w_edge[t][i*K+k][j*K+l]!=INF):
                        ve_index[t][0][i][k][j][l] = cnt
                        ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                               name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t))
                        ce[cnt] = w_edge[t][i*K+k][j*K+l]
                        expr.add(ve[ve_index[t][0][i][k][j][l]], w_edge[t][i*K+k][j*K+l])
                        expr2.add(ve[ve_index[t][0][i][k][j][l]])
                        cnt = cnt +1
                        
                if t > 0 :
                    #left neighbor (time ) connections
                    for k_prime in range(K):
                        if(w_e[t][i*K + k][k_prime]!=INF):
                            ve_index[t][1][i][k][i][k_prime] = cnt
                            #ve[cnt] = m.addVar(vtype=GRB.BINARY, name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t-1)+","+str(t) )
                            #ce[cnt] = w_e[t][i*K + k][k_prime]
                            cnt = cnt +1

                if t < T-1 :

                    #right neighbor (time) connection
                    for k_prime in range(K, K+K):
                        if(w_e[t][i*K + k][k_prime]!=INF):
                            ve_index[t][2][i][k][i][k_prime-K] = cnt
                            ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                               name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t)+","+str(t+1) )
                            ce[cnt] = w_e[t][i*K + k][k_prime]
                            expr.add(ve[ve_index[t][2][i][k][i][k_prime-K]], w_e[t][i*K + k][k_prime])
                            cnt = cnt +1
            m.addConstr(expr2, GRB.EQUAL, 1, "c_II_"+str(constr))
            constr += 1 
        print '.',
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    
    #if an edge is ON then
    #both the corresponding vertices should be ON
    for t in range(T):
        for i in range(N):
            expr2 = LinExpr()
            for k in range(K):
                #sum all connections to right
                expr  = LinExpr()
                j = (i+1) % N
                for l in range(K):
                    if (ve_index[t][0][i][k][j][l]!=-1):
                        expr.add(ve[ve_index[t][0][i][k][j][l]])
                #equating to Vik
                m.addConstr(expr - vv[(i*K + k)*T +t] , GRB.EQUAL, 0, "c_III_" + str(constr))
                constr += 1

                #sum all connections to left
                expr  = LinExpr()
                j = (i-1) % N
                for l in range(K):
                    if (ve_index[t][0][j][l][i][k]!=-1):
                        expr.add(ve[ve_index[t][0][j][l][i][k]])
                #equating to Vik
                m.addConstr(expr - vv[(i*K + k)*T +t], GRB.EQUAL, 0, "c_III_" + str(constr))
                constr += 1
                
                if t > 0:
                    expr  = LinExpr()
                    for l in range(K):
                        if (ve_index[t-1][2][i][l][i][k]!=-1):
                            expr.add(ve[ve_index[t-1][2][i][l][i][k]])
                    #equating to Vik
                    m.addConstr(expr - vv[(i*K + k)*T +t] , GRB.EQUAL, 0, "c_IV_" + str(constr))
                    constr += 1
                if t < T-1:
                    expr  = LinExpr()
                    for l in range(K):
                        if (ve_index[t][2][i][k][i][l]!=-1):
                            expr.add(ve[ve_index[t][2][i][k][i][l]])
                            expr2.add(ve[ve_index[t][2][i][k][i][l]])
                    #equating to Vik
                    m.addConstr(expr - vv[(i*K + k)*T +t] , GRB.EQUAL, 0, "c_IV_" + str(constr))
                    constr += 1
            if t<T-1:
                m.addConstr(expr2, GRB.EQUAL, 1, "c_II_"+str(constr))
                constr += 1

  
    print "\n\n|Vv| = ", T*N*K
    print "|Ve| = ", cnt
    print "|CONSTRAINTS| = ", constr
    
    return m, vv, ve


def build_model_less_constraint(T, N, K, w, w_edge, w_e):
    #set model
    m = Model("model_1")
    INF = 9999999999
    # #############################################################################
    # ### #### #### #### #### DEFINE VARIABLES AND COSTS  ### ### ### ### ### 
    # #############################################################################
    print '\nDEFINING VARIABLES AND COSTS',
    ve = {}
    ce = {}
    
    # Ce.Ve
    ve_index = -1 * np.ones((T, 3, N,K,N,K))
    cnt = 0

    #conections within a frame
    for t in range(T):
        for i in  range(N):
            for k in range(K):
                j = (i+1)%N
                for l in range(K):
                    if(w_edge[t][i*K+k][j*K+l]!=INF):
                        ve_index[t][0][i][k][j][l] = cnt
                        ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                           name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t))
                        ce[cnt] = w_edge[t][i*K+k][j*K+l] + w[t][i][k]/2
                        cnt = cnt +1

    m.update()
    
    
    # #############################################################################
    # ### #### #### #### #### SET OBJECTIVE FUNCTION  ### ### ### ### ### 
    # #############################################################################
    
    print '\nSETTING OBJECTIVE FUNCTION',

    expr = LinExpr()
    #(I.)connections within one frame
    for t in range(T):
        for i in  range(N):
            for k in range(K):
                for j in range(N):
                    for l in range(K):
                        index = ve_index[t][0][i][k][j][l]
                        if(index!=-1):
                            expr.add(ve[index], ce[index])

    m.setObjective(expr, GRB.MINIMIZE)
    
    # #############################################################################
    # ### #### #### #### #### SET CONSTRAINTS  ### ### ### ### ### 
    # #############################################################################
    print '\nSETTING CONSTRAINTS',
    constr = 0 #counter for constraints

    for t in range (T):
        for i in range(N):
            expr2 = LinExpr()
            for k in range(K):
                j= (i+1)%N
                for l in range(K):
                    if(ve_index[t][0][i][k][j][l]!=-1):
                        expr2.add(ve[ve_index[t][0][i][k][j][l]])

            m.addConstr(expr2, GRB.EQUAL, 1, "c_II_"+str(constr))
            constr += 1


    #if an edge is ON then
    #both the corresponding vertices should be ON
    for t in range(T):
        for i in range(N):
            for k in range(K):
                #sum all connections to right
                expr_left  = LinExpr()
                j = (i+1) % N
                for l in range(K):
                    if (ve_index[t][0][i][k][j][l]!=-1):
                        expr_left.add(ve[ve_index[t][0][i][k][j][l]])
                #equating to Vik
                #sum all connections to left
                expr_right  = LinExpr()
                j = (i-1) % N
                for l in range(K):
                    if (ve_index[t][0][j][l][i][k]!=-1):
                        expr_right.add(ve[ve_index[t][0][j][l][i][k]])
                #equating to Vik
                m.addConstr(expr_left - expr_right, GRB.EQUAL, 0, "c_III_" + str(constr))
                constr += 1
    print "\n\n|Vv| = ", T*N*K
    print "|Ve| = ", cnt
    print "|CONSTRAINTS| = ", constr
    
    return m, ve, ve_index


def test_model_without_edge(T, N, K, w):
    #set model
    m = Model("model_2")
    INF = 9999999999
    # #############################################################################
    # ### #### #### #### #### DEFINE VARIABLES AND COSTS  ### ### ### ### ### 
    # #############################################################################
    print '\nDEFINING VARIABLES AND COSTS',
    vv = {}
    cv = {}
    
    # Cvt.Vvt
    for t in range(T):
        for i in range(N):
            for k in range(K):
                vv[(i*K+k)*T +t] = (m.addVar(vtype=GRB.BINARY, name="v " + str(i)+","+str(k) + ","+str(t)))
                cv[(i*K+k)*T +t] = w[t][i][k]
    m.update()
    
    # #############################################################################
    # ### #### #### #### #### SET OBJECTIVE FUNCTION  ### ### ### ### ### 
    # #############################################################################
    
    print '\nSETTING OBJECTIVE FUNCTION',
    expr = LinExpr()

    #expr.addTerms(cv,vv)
    for t in range(T):
        for i in range(N):
            for k in range(K):
                expr.add(vv[(i*K+k)*T +t], w[t][i][k])


    m.setObjective(expr, GRB.MINIMIZE)
    
    # #############################################################################
    # ### #### #### #### #### SET CONSTRAINTS  ### ### ### ### ### 
    # #############################################################################
    print '\nSETTING CONSTRAINTS',
    constr = 0 #counter for constraints

    #exactly one node should be ON per column
    for t in range(T):
        for i in range(N):
            expr1 = LinExpr()
            for k in range(K):
                expr1.add(vv[(i*K + k)*T + t])
            m.addConstr(expr1, GRB.EQUAL, 1, "c_I_"+ str(constr))
            constr += 1            

    
    print "\n\n|Vv| = ", T*N*K
    
    print "|CONSTRAINTS| = ", constr
    
    return m, vv


def test_model_without_temporal(T, N, K, w, w_edge):
   
     #set model
    m = Model("model_1")
    INF = 9999999999
    # #############################################################################
    # ### #### #### #### #### DEFINE VARIABLES AND COSTS  ### ### ### ### ### 
    # #############################################################################
    print '\nDEFINING VARIABLES AND COSTS and SETTING OBJECTIVE FUNCTION',
    ve = {}
    ce = {}
    
    # Ce.Ve
    ve_index = -1 * np.ones((T, 3, N,K,N,K))
    cnt = 0
    constr = 0
    expr = LinExpr()
    #conections within a frame
    for t in range(T):
        for i in  range(N):
            expr2 = LinExpr()
            #expr3 = LinExpr()
            for k in range(K):
                j = (i+1)%N
                for l in range(K):
                    if(w_edge[t][i*K+k][j*K+l]!=INF):
                        ve_index[t][0][i][k][j][l] = cnt
                        ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                           name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t))
                        ce[cnt] = w_edge[t][i*K+k][j*K+l] + w[t][i][k]/2 + w[t][j][l]/2
                        expr.add(ve[cnt], ce[cnt])
                        expr2.add(ve[cnt])
                        cnt = cnt +1
               
        print '.',

    
    #m.update()
    m.setObjective(expr, GRB.MINIMIZE)
    
    
    print '\nSETTING CONSTRAINTS',
     #counter for constraints


    #if an edge is ON then
    #both the corresponding vertices should be ON
    for t in range(T):
        for i in range(N):
            for k in range(K):
                #sum all connections to right
                expr_left  = LinExpr()
                j = (i+1) % N
                for l in range(K):
                    index = ve_index[t][0][i][k][j][l]
                    if (index!=-1):
                        expr_left.add(ve[index])
                #equating to Vik
                #sum all connections to left
                expr_right  = LinExpr()
                j = (i-1) % N
                for l in range(K):
                    index = ve_index[t][0][j][l][i][k]
                    if (index!=-1):
                        expr_right.add(ve[index])
                #equating to Vik
                m.addConstr(expr_left - expr_right, GRB.EQUAL, 0, "c_III_" + str(constr))
                constr += 1
                



  
   

    print "\n\n|Vv| = ", T*N*K
    print "|Ve| = ", cnt
    print "|CONSTRAINTS| = ", constr
    
    return m, ve, ve_index

def build_model_improved_constraint(T, N, K, w, w_edge, w_e):
    #set model
    m = Model("model_1")
    INF = 9999999999
    # #############################################################################
    # ### #### #### #### #### DEFINE VARIABLES AND COSTS  ### ### ### ### ### 
    # #############################################################################
    print '\nDEFINING VARIABLES AND COSTS and SETTING OBJECTIVE FUNCTION',
    ve = {}
    ce = {}
    
    # Ce.Ve
    ve_index = -1 * np.ones((T, 3, N,K,N,K))
    cnt = 0
    constr = 0
    expr = LinExpr()
    #conections within a frame
    for t in range(T):
        for i in  range(N):
            expr2 = LinExpr()
            expr3 = LinExpr()
            for k in range(K):
                j = (i+1)%N
                for l in range(K):
                    if(w_edge[t][i*K+k][j*K+l]!=INF):
                        ve_index[t][0][i][k][j][l] = cnt
                        ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                           name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t))
                        ce[cnt] = w_edge[t][i*K+k][j*K+l] + w[t][i][k]/2 + w[t][j][l]/2
                        expr.add(ve[cnt], ce[cnt])
                        expr2.add(ve[cnt])
                        cnt = cnt +1
                if t < T-1 :

                    #right neighbor (time) connection
                    for k_prime in range(K, K+K):
                        if(w_e[t][i*K + k][k_prime]!=INF):
                            ve_index[t][2][i][k][i][k_prime-K] = cnt
                            ve[cnt] = m.addVar(vtype=GRB.BINARY, 
                                               name="v " + str(i)+","+str(k)+";"+ str(j)+","+str(l)+","+ str(t)+","+str(t+1) )
                            ce[cnt] = w_e[t][i*K + k][k_prime] + w[t][i][k]/2 + w[t+1][i][k_prime-K]/2
                            expr.add(ve[cnt], ce[cnt])
                            expr3.add(ve[cnt])
                            cnt = cnt +1
            m.addConstr(expr2, GRB.EQUAL, 1, "c_II_"+str(constr))
            constr += 1
            if t<T-1:
                m.addConstr(expr3, GRB.EQUAL, 1, "c_II_"+str(constr))
                constr += 1
        print '.',

    
    #m.update()
    m.setObjective(expr, GRB.MINIMIZE)
    
    
    print '\nSETTING CONSTRAINTS',
     #counter for constraints


    #if an edge is ON then
    #both the corresponding vertices should be ON
    for t in range(T):
        for i in range(N):
            for k in range(K):
                #sum all connections to right
                expr_left  = LinExpr()
                j = (i+1) % N
                for l in range(K):
                    index = ve_index[t][0][i][k][j][l]
                    if (index!=-1):
                        expr_left.add(ve[index])
                #equating to Vik
                #sum all connections to left
                expr_right  = LinExpr()
                j = (i-1) % N
                for l in range(K):
                    index = ve_index[t][0][j][l][i][k]
                    if (index!=-1):
                        expr_right.add(ve[index])
                #equating to Vik
                m.addConstr(expr_left - expr_right, GRB.EQUAL, 0, "c_III_" + str(constr))
                constr += 1
                
                if t>0 and t<T-1:
                    expr_left  = LinExpr()
                    for l in range(K):
                        index = ve_index[t-1][2][i][l][i][k] 
                        if (index!=-1):
                            expr_left.add(ve[index])

                    expr_right = LinExpr()
                    for l in range(K):
                        index = ve_index[t][2][i][k][i][l]
                        if (index!=-1):
                            expr_right.add(ve[index])

                    m.addConstr(expr_right - expr_left , GRB.EQUAL, 0, "c_IV_" + str(constr))
                    constr += 1


  
   

    print "\n\n|Vv| = ", T*N*K
    print "|Ve| = ", cnt
    print "|CONSTRAINTS| = ", constr
    
    return m, ve, ve_index



