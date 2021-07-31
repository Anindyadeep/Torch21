'''
*args and **kwargs are genrally used to 
give the dynamic mulriple input to a function
for the execution such that we dont have to change the 
arcguments in the main function everytime, rather than
they would be taken automatically by themselves

NOTE: THE TYPE OF *args --> tuple (non-keyword argument)
      AND THE TYPE OF **kwargs --> dict (keyword argument)
'''

def func1(class_name, *args):
    print(f"The class is {class_name} and students of this class are\n")
    for names in args:
        print(names)

class_name = "45BC3"
func1(class_name, "babin", "Anindya", "Pokemon")

# another way
names = ["pokemon", "digimon"]
func1(class_name, *names) # if we dont pass the * here the it will print the full list at once

'''
(source: GFG)
A keyword argument is where you provide a name to the variable as you pass it into the function.
So we can thinks **kwargs as a dictionary that maps each keyword to the value that we pass alongside it. 
That is why when we iterate over the kwargs there doesn’t seem to be any order in which they were printed out.

i.e. the arguments we pass is a keyword within itself which is having :
                        1. a specific name of it
                        2. that name has a specific value in it

for e.g. let the function is defined as func(**kwargs)
so when we call the function, it will be called like this -->

func(arg1 = 222, arg2 = "pokemon" ....)
and in the function we take those arguements as a dictionary where the keywords 
are the keys and the values of those keywords are the values of the keys
'''

def func2(**kwargs):
    for key, value in kwargs.items():
        print(f"keywords: {key} and value of this keyword is: {value}")

func2(first = 111, second = 222, third = 333)

'''
we can also put the *args and **kwargs at the same time
'''

def competition(competition_name, *first_four, **top_three):
    print("THE NAME OF THE COMPETITION IS: ", competition_name)
    for name in first_four:
        print(name)
    for key, value in top_three.items():
        if key == "first" or key == "second" or key == "third":
            print(f"{key} is {value}")

competition("100 meters", "Anindya", "Subhomoy", "Parthib", "Sudi", first="Anindya", second = "Subhomoy", third = "Parthib")