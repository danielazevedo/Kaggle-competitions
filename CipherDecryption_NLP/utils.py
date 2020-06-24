
import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os
import textwrap

from random import random, sample, shuffle, randint
from operator import itemgetter



def create_cipher():
    letters_1 = list(string.ascii_lowercase)
    letters_2 = list(string.ascii_lowercase)
    true_cypher = {}

    # shuffle
    shuffle(letters_2)

    # create mapping between the letters
    for i, j in zip(letters_1, letters_2):
        true_cypher[j] = i

    return true_cypher





# Function to encode a message
def encode_message(msg, cipher):
	regex = re.compile('[^a-zA-Z]')
	msg = msg.lower()

  	# replace non-alpha characters
	msg = regex.sub(' ', msg)

	# make the encoded message
	coded_msg = []
	for ch in msg:

		#default
		coded_ch = ch
		if ch in cipher:
			coded_ch = cipher[ch]
		coded_msg.append(coded_ch)
	
	return ''.join(coded_msg)


# Function to decode a message
def decode_message(msg, cipher):
  decoded_msg = []
  for ch in msg:

    #default
    decoded_ch = ch
    if ch in cipher:
      decoded_ch = cipher[ch]
    decoded_msg.append(decoded_ch)

  return ''.join(decoded_msg)







class EvolutionaryAlgorithm:
  def __init__(self, n_pop, prob_mutation, prob_crossover, elitism_size, lm):
    
    self.n_pop = n_pop
    self.prob_mutation = prob_mutation
    self.prob_crossover = prob_crossover
    self.elitism_size = elitism_size
    self.best_map = None
    self.model = lm


  def init_population(self, n):

    population = []
    for _ in range(n):
      indv = list(string.ascii_lowercase)
      shuffle(indv)
      population.append(indv)

    return population
  

  #One point Crossover
  def crossover(self, indv_1, indv_2):
    
    if random() < self.prob_crossover:
        
        pos = randint(0,len(indv_1))
        f1 = indv_1[0:pos] + indv_2[pos:]
        f2 = indv_2[0:pos] + indv_1[pos:]
        return [f1, f2]
        
    else:
      return [indv_1,indv_2]


  def swap_mutation(self, indv):
    new_indv = indv.copy()

    if  random() < self.prob_mutation:
      i = np.random.randint(len(new_indv))
      j = np.random.randint(len(new_indv))

      new_indv[i], new_indv[j] = new_indv[j], new_indv[i]

    return new_indv
    

  def elitism(self, population, offspring):
    
    comp_elite = int(len(population) * self.elitism_size)
    
    offspring.sort()
    population.sort()
    
    new_population = population[:comp_elite] + offspring[:len(population) - comp_elite]
    
    return new_population


  def evaluate_population(self, population, original_letters):

    scores = {}
    for indv in population:

      current_map = {}
      for k, v in zip(original_letters, indv):
        current_map[k] = v

      decoded_message = decode_message(encoded_message, current_map)
      score = self.model.compute_sequence_prob(decoded_message)

      scores[''.join(indv)] = [score, current_map]
      
    return scores


  def evolve(self):

    num_iters = 700

    scores = np.zeros(num_iters)
    best_indv = None
    best_score = float('-inf')

    population = self.init_population(self.n_pop)

    for it in range(num_iters):
      

      pool = population
      
      #Crossover
      parents = []
      for i in  range(0, self.n_pop - 1, 2):
          indiv_1= pool[i]
          indiv_2 = pool[i+1]
          parents.extend(self.crossover(indiv_1, indiv_2)) 
      

      #Mutation
      offspring = []
      for indv in parents:
        offspring.append(self.swap_mutation(indv))


      #Select the best inviduals: Elitism
      population = self.elitism(population, offspring)

      pop_scores = self.evaluate_population(population, list(string.ascii_lowercase))

      for s in pop_scores:

        if pop_scores[s][0] > best_score:
          best_indv = s
          self.best_map = pop_scores[s][1]
          best_score = pop_scores[s][0]

      scores[it] = np.mean([pop_scores[s][0] for s in pop_scores])

      if it % 100 == 0:
        print("Iteration ", it, ": score:", round(scores[it], 3), "best so far:", round(best_score, 3))
    
    plt.figure(figsize=(12, 8))
    plt.plot(scores)
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Score Evolution')
    plt.show()


