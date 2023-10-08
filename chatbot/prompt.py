import nltk
from nltk.stem import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import pickle

# Download NLTK data if not already present
nltk.download('punkt')

stemmer = LancasterStemmer()

with open(r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\chatbot\intents.json") as file:
    data = json.load(file)

intents = data['intents']
responses_by_tag = {intent['tag']: intent['responses'] for intent in intents}


words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1) if w in wrds else bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# with open("data.pickle", "wb") as f:
#         pickle.dump((words, labels, training, output), f)

# Build the model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         break

class Bot(): 
    def test(self,message):
        user_input = message
        # Tokenize and preprocess the user input
        user_words = nltk.word_tokenize(user_input)
        user_words = [stemmer.stem(word.lower()) for word in user_words]

        # Create a bag of words
        user_bag = [1 if word in user_words else 0 for word in words]

        # Reshape the bag for model prediction
        user_bag = np.array(user_bag).reshape(1, -1)

        # Get the model's prediction
        model_prediction = model.predict(user_bag)[0]

        # Get the index with the highest probability
        predicted_label_index = np.argmax(model_prediction)

        # Get the corresponding label
        predicted_label = labels[predicted_label_index]

        # Get a random response based on the predicted label
        responses = responses_by_tag.get(predicted_label, ["I'm not sure how to respond."])
        model_response = np.random.choice(responses)
        payload ={"message":model_response}
        return payload

