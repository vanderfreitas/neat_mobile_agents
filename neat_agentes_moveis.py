# Dependency:
# pip install neat-python

#from __future__ import print_function
import os
import neat
#import visualize

import pickle # to save the genome

import numpy as np
import random as rd
import neat
import os
#from __future__ import print_function
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Variáveis globais
# Número de agentes 
N = 5
R = 5 #numero de agentes
T = 200 #tempo máximo
h = 0.1     # passo da integracao
RAIO = 20 #raio do robo
LIMITE_X = 50
LIMITE_Y = 50
SPEED_MAX = 0.8 #m/s
PONTO_DE_PARADA = [200, 200]
THETA_MAX = 2*np.pi #angulo máximo de um agente
D_MAX = 25 #distancia máxima detectada pelo sensor


# Variables that are used several times
TWO_PI = 2.0 * np.pi
PI_OVER_TWO = np.pi/2
PI_OVER_FOUR = np.pi/4
TWO_RADIUS = 2*RAIO
TWO_R = 2*R
ttt = np.arange(0, T, h)
RtimesT = R*T
THETA_DOT_MAX = 0.6*h
INDEX = np.arange(0,R*4,4)




# Controling output figs
print_percent = 200
noffe = 0



"""###Sensor"""

#Parametros: 
#Vetor(lista) u = [x_1, y_1, theta_1, s_1, x_2, y_2, theta_2,s_2 ..., x_n, y_n, theta_n, s_n] de namanho 3n
#indice do agente de referência

# Função para deixar um angulo entre 0 e 2pi
def rot_ang(theta): 
  while (theta < 0):
    theta +=  TWO_PI;
  while (theta >= TWO_PI):
    theta -= TWO_PI; 
  return theta

def rot_angs(beta):
  for i in range(len(beta)):
    beta[i] = rot_ang(beta[i])
  return beta

#Calcula e insere distancia entre sensores na posição referente ao octante
def octantes(alpha, beta, angulos, dx, dy, Dmax, raio):
  octan = -1
  dist = np.sqrt((np.square(dx) + np.square(dy)))
  d_octa = [Dmax]*8 #vetor de distancias (para cada octante)
  #verificando octante
  for k in range(len(alpha)):
    for i in range(8):
      if (dist[k] - TWO_RADIUS <=Dmax):  
        if ((angulos[i] < angulos[i+1])):
          if (((alpha[k]) >= angulos[i]) and ((alpha[k]) < angulos[i+1]) and (d_octa[i] > dist[k] - TWO_RADIUS)):
            #print(f'colocou a dist {np.around(dist[k]- 2*raio,2)} em {i+1}')
            d_octa[i] = dist[k] - TWO_RADIUS
        else: 
          if (((alpha[k]) >= (angulos[i]-TWO_PI)) and ((alpha[k]) < angulos[i+1]) and (d_octa[i] > dist[k] - TWO_RADIUS)):
            #print(f'colocou a dist {np.around(dist[k]- 2*raio,2)} em {i+1}')
            d_octa[i] = dist[k] - TWO_RADIUS
  return d_octa 

def sensor(u, idd, Dmax, raio, P):
 
  # Referencia: agente idd
  # Agente vizinho (neighbor): agente n
  x_i = u[INDEX[idd]] #Cordenada x do agente de referencia
  y_i = u[INDEX[idd] + 1] #Cordenada y agente de referencia
  theta_agent_i = u[INDEX[idd]+ 2]  #Theta do agente de referencia (em radianos)
  #max = 25.0
  #r = 14.08

  #Deixando o angulo do agente de referencia entre 0 e 2pi
  theta_agent_i = rot_ang(theta_agent_i)

  # Computando os dx, dx, para encontrar o angulo da diferenca entre as posicoes dos agentes
  # e o respectivo quadrante do vizinho. Esse eh um passo inicial e o quadrante encontrado ainda 
  # nao eh o final, pois sera preciso rotacionar o sistema, de acordo com o theta do agente de referencia
  #x_n = -10.0
  #y_n = -20.0
  N = int(len(u)/4)
  dx = [0]*(N-1)
  dy = [0]*(N-1)
  theta = [0]*N

  #Pegando distancias dx e dx do vetor u 
  for k in range(N-1):
    if (k >= idd):
      dx[k] = u[INDEX[k+1]] - x_i
      dy[k] =  u[INDEX[k+1] + 1]- y_i
      theta[k] = u[INDEX[k] + 2]
    if (k < idd):
      dx[k] =  u[INDEX[k]] - x_i
      dy[k] =  u[INDEX[k] + 1]- y_i
      theta[k] = u[INDEX[k] + 2]

  #Vetor com intervalos para representar os octantes 
  #variando de (theta) a (theta + 2*pi) de pi/4 em pi/4
  angulos = [0]*9
  angulos[0] = theta_agent_i - PI_OVER_TWO
  angulos[0] = rot_ang(angulos[0])
  for i in range(8):
    angulos[i+1] =  angulos[i] + PI_OVER_FOUR
    # deixando os angulos de cada octante entre 0 e 2pi
    angulos[i+1] = rot_ang(angulos[i+1])

  for i in range(9):
    if (i>0 and angulos[i]==0):
      angulos[i] = TWO_PI

  octan = -100
  alpha = np.arctan(np.divide(dy,dx))
  alpha = rot_angs(alpha)

  #ajustando angulos alpha
  for i in range(len(alpha)): 
    if (dy[i] < 0):
      alpha[i] += np.pi

  #print(f'angulos = {np.around(angulos, 2)}')
  #print(f'alpha = {np.around(alpha, 2)}')
  #quatro primeiros octantes
  beta =  theta_agent_i - alpha
  #quatro ultimos octantes
  beta_pi  = beta - np.pi 
  # deixando os angulos de cada octante entre 0 e 2pi
  beta = rot_angs(beta)
  beta_pi = rot_angs(beta_pi)
  
  Dx = P[0] - x_i
  Dy = P[1] - y_i
  D = np.sqrt( np.square(Dx) + np.square(Dy)) #Distancia entre o agente idd e o ponto de parada
  gama = np.arctan(Dy/Dx)
  ang_idd = theta_agent_i - gama
  ang_idd = rot_ang(ang_idd) #angulo entre o agente idd e ponto de parada

  retorno = octantes(alpha, beta, angulos, dx, dy, Dmax, raio)
  retorno.append(D) #Distando do agente idd ao ponto de parada
  retorno.append(ang_idd) #angulo entre do agente idd e o ponto de parada
  retorno.append(1) #bias
  #retorno =  adicionar alpha e D e o bias = 1
  return retorno
#  return distancia(beta, angulos, dx, dy)

"""###Distâncias - f-homming"""

#funções de distancia 

#Calcula distancia euclidiana
def distancia(x, y):
	dist = np.sqrt(np.square(x) + np.square(y))
	return dist

#Calcula vetor de distancias iniciais de cada agente ao ponto de pararada
def distancia_inicial(u, P, R, RAIO):
	inicialDist = [0]*R #distancia inicial do robo ao ponto de parada
	for r in range(R):
	  inicialDist[r] = distancia(u[0][INDEX[r]]-P[0], u[0][INDEX[r]+1]-P[1]) - TWO_RADIUS
	return inicialDist

#Calcula distancia em um tempo t de cada agente ao ponto de parada
def distParada(u, P, R, t):
  distr=[0]*R
  for r in range(R):
    x = u[t][INDEX[r]] - P[0]
    y = u[t][INDEX[r] +1] - P[1]
    distr[r] = distancia(x,y) - TWO_RADIUS
  return distr

#Calcula distancia mínima entre os agentes num tempo t
def min_dist(u, t, RAIO, R):
	dx = u[t][0] - u[t][0]
	dy = u[t][1] - u[t][1]
	minDist = np.sqrt(np.square(dx) + np.square(dy)) - TWO_RADIUS
	for i in range(R):
		for j in range(R):
			dx = u[t][INDEX[i]] - u[t][INDEX[j]]
			dy = u[t][INDEX[i]+1] - u[t][INDEX[j]+1]
			dist = distancia(dx,dy) - TWO_R
			if (dist < minDist):
				minDist = dist
	return minDist

def f_homming(u):
  global R, RAIO

  P = PONTO_DE_PARADA
  #Inicializa o valor de f_homing = 0.0
  fh = 0.0
	# Verificar a menor distancia entre quaisquer pares de robos no instante zero
  minDist = min_dist(u, 0, RAIO, R)
  #startingDist de cada robo
  for t in range(len(u)):
    #dist = [0]*T
    #inicialDist = [1]*T
    # Atualizar minDist neste instante t
    minDist = min_dist(u, t, RAIO, R)
    dist = distParada(u, P, R, t)
    inicialDist = distancia_inicial(u, P, R, RAIO)
    for r in range(R): # para cada robo
      #print(f'distancias: {inicialDist[r]}, {dist[r]}')
      fh += (inicialDist[r] - dist[r]) / inicialDist[r]
  fh = fh / (RtimesT)
  S =0.1 +  (max(0, min(5,minDist))*0.9)/5
  fh = fh * S
  return fh

"""###Gerador de posições iniciais"""

def distancia(x, y):
	dist = np.sqrt(np.square(x) + np.square(y))
	return dist

def pos_partida(R, LIMITE_X, LIMITE_Y, THETA_MAX, SPEED_MAX, RAIO): 
  u = [0]*(R*4)
  u[0] = rd.random()*LIMITE_X  # x
  u[1] = rd.random()*LIMITE_Y  # y
  u[2] = rd.random()*THETA_MAX #aleatorio de 0 a 2*pi
  u[3] = rd.random()*SPEED_MAX #velocidade aleatoria

  for r in range(1,R):
    u[INDEX[r]] = rd.random()*LIMITE_X #x
    u[INDEX[r]+1] =rd.random()*LIMITE_Y  # y
    dist = [3*RAIO]*R
    '''for i in range(r):
      dist[i] = distancia(u[r*4]- u[i*4], u[r*4+1]-u[i*4+1])
    while (min(dist) < TWO_RADIUS):
      u[r*4] = rd.random()*LIMITE_X #x
      u[r*4+1] =rd.random()*LIMITE_Y  # y
      dist = [3*RAIO]*R
      for i in range(r):
        dist[i] = distancia(u[r*4]- u[i*4], u[r*4+1]-u[i*4+1])'''

    u[INDEX[r]+2] =rd.random()*TWO_PI #aleatorio de 0 a 2*pi
    u[INDEX[r]+3] = rd.random()*SPEED_MAX #velocidade aleatoria
  return u

"""Salvando e imprimindo trajetoria"""

def Trajectory(u, t):
  f = open('trajectory.csv', 'w')

  for i in range(0,len(t),10):
    for idd in range(N):
      f.write(str(t[i]))
      x = u[idd*4][i]
      y = u[idd*4+1][i]
      theta = u[idd*4+2][i]
      s = u[idd*4+3][i]
      f.write(';' + str(idd+1) + ';' + str(x) + ';' + str(y) + ';' + str(theta) + ';' + str(s) +'\n')

  f.close()

def printTrajetory(u):
  global noffe
  # Para cada particula, imprimir suas coordenadas
  # u[i*3]    eh x
  # u[i*3+1]  eh y
  # u[i*3+2]  eh theta
  for i in range(R):
    plt.plot(u[INDEX[i]], u[INDEX[i]+1], label=str(i))

  # destino (ponto de parada)
  plt.scatter(PONTO_DE_PARADA[0], PONTO_DE_PARADA[1], marker='X', color='black', s=50)

  plt.legend(loc='best')
  plt.ylabel('y')
  plt.xlabel('x')
  plt.grid()
  #plt.show()
  plt.savefig('figs/' + str(noffe) + '_fig.png')
  plt.clf()

"""Neat + simulação"""




#Função model
def model(t, u, N, net):
  #global noffe

  dudt = []
  #Para cada agente 
  for i in range(N):
    #simula os sensores
    octantes_new = sensor(u, i, D_MAX, RAIO, PONTO_DE_PARADA)
    
    net_output = net.activate(octantes_new)
    #if (noffe % print_percent == 0) and (i == 0) and (int(t) == 200):
    #  print(net_output)

    # ANN output (angular velocity and acceleration)
    theta_dot = net_output[0]
    s_dot = net_output[1]

    # Preventing the robot physical limits to be trespassed
    theta_agent_i = u[INDEX[i] + 2]
    s_i  = u[INDEX[i] + 3]

    # max angular velocity
    if theta_dot > THETA_DOT_MAX:
      theta_dot = THETA_DOT_MAX
    elif theta_dot < -THETA_DOT_MAX:
      theta_dot = -THETA_DOT_MAX

    # max speed
    if s_i + s_dot > SPEED_MAX:
      s_dot = 0.0
    elif s_i + s_dot < 0:
      s_dot = 0.0

    #determina as derivadas com a saida da rede
    dudt.append(u[INDEX[i]+3]*np.cos(u[INDEX[i]+2]))
    dudt.append(u[INDEX[i]+3]*np.sin(u[INDEX[i]+2]))
    dudt.append(theta_dot)
    dudt.append(s_dot)
  return dudt

def simulacao(net): 
  global noffe, T, N, R, LIMITE_X, LIMITE_Y, THETA_MAX, SPEED_MAX, RAIO

	#Determina posições iniciais dos agentes dentro de uma area delimitada
  #por LIMITE_X e LIMITE_Y
  u_part = pos_partida(R, LIMITE_X, LIMITE_Y, THETA_MAX, SPEED_MAX, RAIO)

  t = ttt
  sol = solve_ivp(fun=model, t_span=[0, T], y0=u_part, args=(N, net), t_eval=t, dense_output=True, first_step=h)
  #tempo de integracao: esta funcao gera uma sequencia que inicia em zero e vai ate tf, com passo h
  u = sol.sol(t)
  f = f_homming(u)

  if (noffe % print_percent == 0) or (f > 20.0) or (f < -10):
    #Preenhe arquivo trajectory
    #Trajectory(u, t)
    #Mostra trajetoria
    printTrajetory(u)
  # Number of fitness function evaluations
  print(f"noffe = {noffe} | f_homming = {f}")
  noffe += 1
  
  return f


def eval_genomes(genomes, config):
  for genome_id, genome in genomes:
    #genome.fitness = 0.0

    #cria uma rede (como eh a mesma para todos, cria-se apenas uma vez)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #octantes_new = sensor(u, 0, D_MAX, RAIO, PONTO_DE_PARADA)
    genome.fitness = simulacao(net)
    #print(genome_id)

def run(config_file):
  # Load configuration.
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_file)

  # Create the population, which is the top-level object for a NEAT run.
  p = neat.Population(config)

  # Add a stdout reporter to show progress in the terminal.
  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)
  p.add_reporter(neat.Checkpointer(20))

  # Run for up to 300 generations.
  winner = p.run(eval_genomes, 2000)

  # Save the winner.
  with open('winner-network', 'wb') as f:
    pickle.dump(winner, f)

  # Display the winning genome.
  print('\nBest genome:\n{!s}'.format(winner))

  #visualize.draw_net(config, winner, True)
    
if __name__ == '__main__':
  # Determine path to configuration file. This path manipulation is
  # here so that the script will run successfully regardless of the
  # current working directory.
  local_dir = os.path.dirname('__file__')
  config_path = os.path.join(local_dir, 'config-feedforward')
  run(config_path)








# DEPOIS QUE CONSEGUIR A REDE IDEAL, PARA CARREGÁ-LA, BASTA FAZER:
# load the winner
#with open('winner-network', 'rb') as f:
#  net = pickle.load(f)