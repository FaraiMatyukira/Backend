import random
from random import randint
class tools:

    def generate_id(self):
        try:
            id = ""
            for i in range (0,12):
                number  = randint(0,9)
                id += number 

            return id 
                
        except Exception as e  :
            print("ERROR on tool(generate_id)",e)
