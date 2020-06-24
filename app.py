
from flask import Flask, render_template, request, Markup
from tabulate import tabulate
import string

import ArticleSpinner_NLP.utils as article_sp
import CipherDecryption_NLP.utils as cipher_func
import FIFA_20_analysis.utils as fifa20_func

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import keras
from keras.models import load_model
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc

from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly.graph_objects as go

app = Flask(__name__)




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
            random.shuffle(indv)
            population.append(indv)

        return population


  	#One point Crossover
    def crossover(self, indv_1, indv_2):

        if random.random() < self.prob_crossover:

            pos = randint(0,len(indv_1))
            f1 = indv_1[0:pos] + indv_2[pos:]
            f2 = indv_2[0:pos] + indv_1[pos:]
            return [f1, f2]

        else:
            return [indv_1,indv_2]


    def swap_mutation(self, indv):
        new_indv = indv.copy()

        if  random.random() < self.prob_mutation:
            i = np.random.randint(len(new_indv))
            j = np.random.randint(len(new_indv))

            new_indv[i], new_indv[j] = new_indv[j], new_indv[i]

        return new_indv


    def elitism(self, population, offspring):

        comp_elite = int(len(population) * self.elitism_size)

        pop_scores = self.evaluate_population(population, list(string.ascii_lowercase))
        offsprings_scores = self.evaluate_population(offspring, list(string.ascii_lowercase))
        
        pop_scores = [[k, pop_scores[k][0]] for k in pop_scores]
        offsprings_scores = [[k, offsprings_scores[k][0]] for k in offsprings_scores]

        
        pop_scores.sort(key=lambda x: x[1])
        offsprings_scores.sort(key=lambda x: x[1])
        
        population_inds = pop_scores[::-1][:comp_elite]
        offspring_inds = offsprings_scores[::-1][:len(population) - comp_elite]
        

        new_population = [list(i[0]) for i in population_inds] + [list(j[0]) for j in offspring_inds]
        
        #print(population)
        #print(len(offspring))
        #print(new_population)
        return new_population

    
    def evaluate_population(self, population, original_letters):

        scores = {}
        for indv in population:
            current_map = {}
            for k, v in zip(original_letters, indv):
                current_map[k] = v

            decoded_message = cipher_func.decode_message(encoded_message, current_map)
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




### Create the Language Model class: N-Gram using Markov Assumption
class LanguageModel:
  def __init__(self, ngram, pi):
    self.ngram = ngram
    self.pi = pi


  #Compute the ngram probabilities
  def train(self, text_name):

    # for replacing non-alpha characters
    regex = re.compile('[^a-zA-Z]')

    # load in words
    for line in open(text_name):
      line = line.rstrip()

      # exclude blank lines
      if line:
        # replace all non-alpha characters with space
        line = regex.sub(' ', line) 

        # split the tokens in the line and lowercase
        words = line.lower().split()


        for word in words:
          
          # first letter
          ch0 = word[0]
          self.update_pi(ch0)

          # other letters
          for ch1 in word[1:]:
            self.update_probabilities(ch0, ch1)
            ch0 = ch1

    # normalize the probabilities
    self.pi /= self.pi.sum()
    self.ngram /= self.ngram.sum(axis=1, keepdims=True)


  # Compute Initial probabilities
  def update_pi(self, ch):
    i = ord(ch) - 97
    self.pi[i] += 1


  def update_probabilities(self, ch1, ch2):
    # ord('a') = 97
    
    i = ord(ch1) - 97
    j = ord(ch2) - 97
    
    self.ngram[i,j] += 1


  # Compute the log-probability of a word
  def compute_word_prob(self, word):

    i = ord(word[0]) - 97
    logp = np.log(self.pi[i])

    for ch in word[1:]:
      j = ord(ch) - 97
      logp += np.log(self.ngram[i, j])
      i = j

    return logp


  # Compute the probability of a sequence of words
  def compute_sequence_prob(self, words):
    
    #split string into words
    if type(words) == str:
      words = words.split()

    logp = 0
    for word in words:
      logp += self.compute_word_prob(word)
    return logp


encoded_message = ''
true_cipher = {}

@app.route('/')
def index():
   return render_template('home.html');





###### ARTICAL SPINNER
@app.route("/art_spinner", methods=['POST'])
def article_spinner():
    ngram = pickle.load(open('ArticleSpinner_NLP/article_spinner_model.pickle', "rb"))
    new_text = article_sp.test_spinner(request.form['text'], ngram)

    return render_template('compare_texts.html', old_text=request.form['text'], forward_message=new_text);


@app.route("/redirect_text_form", methods=['POST'])
def redirect_text_form():
    return render_template('text_form.html');





###### CIPHER DECRYPTION

@app.route("/redirect_cipher_form", methods=['POST'])
def redirect_cipher_form():
	global true_cipher
	true_cipher = cipher_func.create_cipher()
	print(true_cipher)

	row1 = list(string.ascii_lowercase)
	row2 = [true_cipher[i] for i in row1]

	tt = tabulate([row1, row2], tablefmt='html')

	return render_template('cipher_input.html', tt= Markup(tt))


@app.route("/cyper_decr", methods=['POST'])
def cyper_decryption():
	global encoded_message

	l_model = pickle.load(open('CipherDecryption_NLP/lm.pickle', "rb"))
	encoded_message = cipher_func.encode_message(request.form['text'], true_cipher)
	
	
	ea = EvolutionaryAlgorithm(n_pop = 30,
	                           prob_mutation = 1,
	                           prob_crossover = 0,
	                           elitism_size = 0.6,
	                           lm = l_model)

	ea.evolve()


	decoded_message = cipher_func.decode_message(encoded_message, ea.best_map)

	return render_template('compare_texts_cipher.html', old_text=encoded_message, forward_message=decoded_message);



#### CREDIT CARD FRAUD

@app.route("/crd_fraud", methods=['POST'])
def model_crd_fraud():

	#input
	[X_train, X_test, y_test] = pickle.load(open('CreditCardFraud/cc_data.pickle', "rb"))
	X_train.reset_index(inplace=True, drop=True)


	#results
	svm_model = pickle.load(open('CreditCardFraud/cc_SVM.pickle', "rb"))

	y_pred = svm_model.predict(X_test)
	y_pred = (y_pred > 0.5)
	
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	roc_score = auc(fpr, tpr)

	fig = go.Figure([
	    Scatter(x=fpr, y=tpr, name="ROC curve (area = " + str(round(roc_score, 2)) + ")")
	])

	fig.update_layout(
	    title="ROC curve",
	    xaxis=dict(title="False Positive Rate"),
	    yaxis=dict(title="True Positive Rate"),
	    autosize=False,
	    width=1200,
	    height=700,
	    showlegend=True
	)


	my_plot_div = plot(fig, output_type="div")

	return render_template('credit_card.html', df = Markup(X_train.head(10).to_html()), results = Markup(my_plot_div))


@app.route("/digit_recog", methods=['POST'])
def model_digit_recg():

	#random digit
	[digits, label] = pickle.load(open('Digit Recognizer/digits.pickle', "rb"))

	true_digit = random.choice(digits)
	plt.imshow(true_digit.reshape([28,28]), cmap='binary')
	plt.savefig('static/digit_test.png')


	classifier = load_model('Digit Recognizer/classifier.h5')
	digit_prediction = classifier.predict(true_digit.reshape([1, 28,28, 1]))
	digit_prediction = np.argmax(digit_prediction)
	

	return render_template('digit_recog.html', digit_prediction = digit_prediction);


nations = []
formations = ['352', '433', '442']
default_team = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

@app.route("/fifa20", methods=['POST'])
def menu_options():
	global fifa_20, nations
	fifa_20 = pickle.load(open('FIFA_20_Analysis/fifa_data.pickle', "rb"))
	nations = list(fifa_20['nationality'].unique())
	return render_template('menu_fifa20.html');	


@app.route("/best_team", methods=['POST'])
def best_team():
	return render_template('best_team_fifa20.html', nations = nations, formations=formations, best_team = default_team);


@app.route("/get_team", methods=['POST'])
def get_team():
	global nations, formations

	tactic_433 = {
	        'name': "433",
	        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
	        }
	    
	tactic_442 = {
	        'name': "442",
	        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'CDM', 'RM', 'ST', 'ST']
	        }
	    
	tactic_352 = {
	        'name': "352",
	        'positions': ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
	        }

	ind = nations.index(request.form['nation'])
	nations[ind], nations[-1] = nations[-1], nations[ind]

	ind = formations.index(request.form['tactic'])
	formations[ind], formations[-1] = formations[-1], formations[ind]

	if formations[-1] == '433':
		t = tactic_433
	elif formations[-1] == '442':
		t = tactic_442
	else:
		t = tactic_352

	best_team = fifa20_func.get_best_team(fifa_20, nations[-1], t)
	if len(best_team) == 0:
		best_team = default_team

	return render_template('best_team_fifa20.html', nations = nations, formations = formations, best_team = best_team);
	


@app.route("/wonderkids", methods=['POST'])
def get_wonderkids():
	players = fifa20_func.get_best_talents(fifa_20, 80)
	return render_template('wonderkids.html', players = Markup(players.to_html()));


@app.route("/positions", methods=['POST'])
def get_positions():
	#fifa20_func.positions_distribution(fifa_20, ['GK', 'DEF', 'MED', 'ATA'])
	#fifa20_func.all_positions_distribution(fifa_20)
	return render_template('relationship_positions.html');


@app.route("/sms", methods=['POST'])
def get_sms_view():
	return render_template('sms_view.html', spam_message = '');


@app.route("/sms_prediction", methods=['POST'])
def get_sms_spam():
	
	sms_model_count = pickle.load(open('SMS_Spam_Detector_NLP/model_count_spam.pickle', "rb"))
	sms_model = pickle.load(open('SMS_Spam_Detector_NLP/model_spam.pickle', "rb"))
	result = sms_model.predict(sms_model_count.transform(np.array([request.form['text']])))
	if result[0] == 0:
		spam_message = 'Not a Spam Message'
	elif result[0] == 1:
		spam_message = 'It is a Spam Message'

	return render_template('sms_view.html', spam_message = spam_message);


@app.route("/sentiment", methods=['POST'])
def get_sentiment_view():
	return render_template('sentiment_view.html', spam_message = '');

@app.route("/sentiment_prediction", methods=['POST'])
def get_sentiment():
	sms_model_count = pickle.load(open('SentimentAnalysis_NLP/model_sentiment_count.pickle', "rb"))
	sms_model = pickle.load(open('SentimentAnalysis_NLP/model_sentiment.pickle', "rb"))
	result = sms_model.predict(sms_model_count.transform(np.array([request.form['text']])))
	
	if result[0] == 0:
		sent_msg = 'Negative Review!'
	elif result[0] == 1:
		sent_msg = 'Positive Review!'


	return render_template('sentiment_view.html', spam_message = sent_msg);



if __name__ == '__main__':
   app.run(debug = True)