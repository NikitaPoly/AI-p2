'''
Decision Tree Induction
Starter code author: Steven Bogaerts
'''

import math

# Copy your AttributeSpec, Example, and ExampleList classes into this file
class AttributeSpec:
    def __init__(self,name,vals):
        self.__name = name
        self.__vals = vals
    def getName(self):
        return self.__name
    def getValAt(self,i):
        return self.__vals[i]
    def getIndexOf(self,value):
        return self.__vals.index(value)
    def getNumVals(self):
        return len(self.__vals)
    def __repr__(self):
        return "AttributeSpec{" + str(self.__name) + ", " + str(self.__vals) + "}"
def testAttributeSpec():
    print("==================== testAttributeSpec ====================")
    tempSpec = AttributeSpec("Temperature", ["Low", "Medium", "High"])
    ratingSpec = AttributeSpec("Rating", [0, 1, 2, 3, 4, 5])
    
    print(tempSpec)
    print(ratingSpec)
    
    print(tempSpec.getName(), tempSpec.getValAt(1), tempSpec.getIndexOf("Medium"), tempSpec.getNumVals())
class Example(object):
    '''
    Example is an example - a collection of attribute keys and values,
    and possibly the class of the example, if known.

    cls - the class of this example (can be left off if unknown)

    nameValDict - dictionary from attribute name (from an AttributeSpec) to the
                 corresponding value for this Example.
    '''

    def __init__(self, nameValDict, cls=None):
        self.__cls = cls
        self.__nameValDict = nameValDict
    def getClass(self):
        return self.__cls
    def hasSameClassAs(self,other):
        return self.__cls == other.__cls
    def getValFor(self,attributeName):
        return self.__nameValDict.get(attributeName)
    def __repr__(self):
        return "Example{"+ self.__cls + ", " + str(self.__nameValDict) + "}"
def testExample():
    print("==================== testExample ====================")
    # "Do you need help lifting this thing?"
    ex1 = Example({"height":5, "width":4, "isHeavy":False}, "No")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}) # cls not provided, so it's None (see __init__)
    
    print(ex1)
    print(ex1.getClass())
    print(ex1.hasSameClassAs(ex2))
    print(ex1.getValFor("width"))   
class ExampleList(object):
    '''
    A list of Example objects.
    '''

    def __init__(self, exampleList):
        self.__exampleList = exampleList
    def __repr__(self):
        return "ExampleList{" + str(self.__exampleList) + "}"
    def getExampleAt(self, i):
        return self.__exampleList[i]
    def getNumExamples(self):
        return len(self.__exampleList)
    def isEmpty(self):
        return self.__exampleList == []
    def append(self, ex):
        self.__exampleList.append(ex)
    def allSameClass(self):
        standardCLS = self.__exampleList[0]
        for example in self.__exampleList:
            if not standardCLS.hasSameClassAs(example):
                return False
        return True
        def specialLog2(self, val):
        if val == 0:
            return 0
        else:
            return math.log(val, 2)

    def calcEntropy(self, classAttrSpec):
        # Calculates the entropy of this example list over the class.
        return -1
        
    def calcGain(self, attrSpec, classAttrSpec):
        # Calculates the gain from splitting on attribute attrSpec.
        return -1

    def chooseAttr(self, attrSpecs, classAttrSpec):
        # Returns the attribute specification that best splits the example list.
        return None

    def countClassValues(self, classAttrSpec):
        '''
        For the examples in this ExampleList,
        counts the number of occurrences of each class.
        Returns a list valCount, where valCount[i] is the number of occurrences of the ith class value
        (as ordered in classAttrSpec).
        '''
        # Initialize valCount
        valCount = [0 for i in range(classAttrSpec.getNumVals())]
        ''' The above line (a "list comprehension") does the same as the following:
        valCount = []
        for i in range(classAttrSpec.getNumVals()):
            valCount.append(0)
        '''
        
        for ex in self.__exampleList: 
            for i in range(classAttrSpec.getNumVals()):
                if classAttrSpec.getValAt(i) == ex.getClass():
                    valCount[i] += 1
            
        return valCount
    def getMajorityClass(self, classAttrSpec):
        '''
        Determines the class that describes the majority of examples
        in the list.
        '''
        valCount = self.countClassValues(classAttrSpec)

        maxID = 0
        for i in range(1, len(valCount)):
            if valCount[i] > valCount[maxID]:
                maxID = i
        return classAttrSpec.getValAt(maxID) # change this to what it should be
    def split(self, attrSpec):
        '''
        Splits the examples by attribute.
        Returns a list of ExampleList objects. (So essentially, a list of lists, or a 2-D list.)
        '''
        
        # Initialize the splitExamples structure to a list of empty ExampleList objects.
        splitExamples = [ExampleList([]) for i in range(attrSpec.getNumVals())]
        ''' The above line (a "list comprehension") does the same as the following:
        splitExamples = []
        for i in range(attrSpec.getNumVals()):
            splitExamples.append(ExampleList([]))
        '''
        
        attrName = attrSpec.getName()
        for ex in self.__exampleList: # Using the list for-each notation
            for i in range(attrSpec.getNumVals()):
                if attrSpec.getValAt(i) == ex.getValFor(attrName):
                    splitExamples[i].append(ex)
        
        return splitExamples
def testExampleListBasic():
    print("==================== testExampleListBasic ====================")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":12, "width":7, "isHeavy":True}, "Yes")

    exList = ExampleList([ex2, ex3])
    print("1)", exList)
    print("2)", exList.getExampleAt(1))
    print("3)", exList.getNumExamples())
    print("4)", exList.isEmpty())
    
    exList.append(ex4)
    print("5)", exList)    
def testAllSameClass():
    print("==================== testAllSameClass ====================")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":12, "width":7, "isHeavy":False}, "Yes")
    ex5 = Example({"height":9, "width":3, "isHeavy":False}, "No")
    ex6 = Example({"height":9, "width":3, "isHeavy":True}, "Yes")
    
    exList = ExampleList([ex2, ex3, ex4])
    print(exList.allSameClass())
    exList.append(ex5)
    exList.append(ex6)
    print(exList.allSameClass())
def testCountClassValues():
    print("==================== testCountClassValues ====================")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":12, "width":7, "isHeavy":False}, "Yes")
    ex5 = Example({"height":9, "width":3, "isHeavy":False}, "No")
    ex6 = Example({"height":9, "width":3, "isHeavy":True}, "Yes")

    exList = ExampleList([ex2, ex3, ex4, ex5, ex6])
    classAttr = AttributeSpec("Need Help?", ["No", "Yes"])
    print(exList.countClassValues(classAttr))   
def testGetMajorityClass():
    print("==================== testGetMajorityClass ====================")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":12, "width":7, "isHeavy":False}, "Yes")
    ex5 = Example({"height":9, "width":3, "isHeavy":False}, "No")
    ex6 = Example({"height":9, "width":3, "isHeavy":True}, "Yes")

    exList = ExampleList([ex2, ex3, ex4, ex5, ex6])
    classAttr = AttributeSpec("Need Help?", ["No", "Yes"])
    print(exList.getMajorityClass(classAttr))

    ex7 = Example({"height":9, "width":4, "isHeavy":False}, "No")
    exList2 = ExampleList([ex5, ex6, ex7])
    print(exList2.getMajorityClass(classAttr))
def testSplit():
    print("==================== testSplit ====================")
    ex2 = Example({"height":'S', "width":'L', "isHeavy":True}, "Yes")
    ex3 = Example({"height":'S', "width":'S', "isHeavy":True}, "Yes")
    ex4 = Example({"height":'L', "width":'M', "isHeavy":False}, "Yes")
    ex5 = Example({"height":'M', "width":'S', "isHeavy":False}, "No")
    ex6 = Example({"height":'M', "width":'S', "isHeavy":True}, "Yes")

    exList = ExampleList([ex2, ex3, ex4, ex5, ex6])
    heightSpec = AttributeSpec("height", ['S', 'M', 'L'])
    widthSpec = AttributeSpec("width", ['S', 'M', 'L'])
    isHeavySpec = AttributeSpec("isHeavy", [False, True])
    
    print("----- Split by height")
    splitByHeight = exList.split(heightSpec)
    for exListID in range(len(splitByHeight)):
        print("Height", heightSpec.getValAt(exListID), "has list:", splitByHeight[exListID], end='\n\n')
    
    print("----- Split by isHeavy")
    splitByIsHeavy = exList.split(isHeavySpec)
    for exListID in range(len(splitByIsHeavy)):
        print("isHeavy", isHeavySpec.getValAt(exListID), "has list:", splitByIsHeavy[exListID], end='\n\n')
def testAll():
    testAttributeSpec()
    testExample()
    testExampleListBasic()
    testAllSameClass()
    testCountClassValues()
    testGetMajorityClass()
    testSplit()
testSplit()
'''
# Put the following methods into your ExampleList class:
    
    def specialLog2(self, val):
        if val == 0:
            return 0
        else:
            return math.log(val, 2)

    def calcEntropy(self, classAttrSpec):
        # Calculates the entropy of this example list over the class.
        return -1
        
    def calcGain(self, attrSpec, classAttrSpec):
        # Calculates the gain from splitting on attribute attrSpec.
        return -1

    def chooseAttr(self, attrSpecs, classAttrSpec):
        # Returns the attribute specification that best splits the example list.
        return None
    
'''

############################################################################

def testEntropy():
    classAttr = AttributeSpec("needHelp", ["No", "Yes"])
    
    emptyList = ExampleList([])
    print(emptyList.calcEntropy(classAttr)) # The entropy of an empty list is defined to be 0, as a special case
    
    ex1 = Example({"height":5, "width":4, "isHeavy":False}, "No")
    ex2 = Example({"height":5, "width":12, "isHeavy":True}, "Yes")
    ex3 = Example({"height":5, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":12, "width":7, "isHeavy":True}, "Yes")
    exList = ExampleList([ex1, ex2, ex3, ex4])
    print(exList.calcEntropy(classAttr))
    
    classAttr = AttributeSpec("result", [0, 1, 2])
    ex1 = Example({"a":2}, 1)
    ex2 = Example({"a":3}, 1)
    ex3 = Example({"a":5}, 2)
    ex4 = Example({"a":4}, 2)
    ex5 = Example({"a":1}, 0)
    exList = ExampleList([ex1, ex2, ex3, ex4, ex5])
    print(exList.calcEntropy(classAttr))
    
def testGain():
    heightAttr = AttributeSpec("height", [1, 2, 3])
    widthAttr = AttributeSpec("width", [1, 2, 3])
    heavyAttr = AttributeSpec("isHeavy", [False, True])
    classAttr = AttributeSpec("needHelp", ["No", "Yes"])
    ex1 = Example({"height":1, "width":1, "isHeavy":False}, "No")
    ex2 = Example({"height":1, "width":3, "isHeavy":True}, "Yes")
    ex3 = Example({"height":1, "width":2, "isHeavy":True}, "Yes")
    ex4 = Example({"height":3, "width":3, "isHeavy":True}, "Yes")
    ex5 = Example({"height":2, "width":2, "isHeavy":False}, "No")
    exList = ExampleList([ex1, ex2, ex3, ex4, ex5])
    
    print(exList.calcGain(heightAttr, classAttr))
    print(exList.calcGain(widthAttr, classAttr))
    print(exList.calcGain(heavyAttr, classAttr))

def testChooseAttr():
    heightAttr = AttributeSpec("height", [1, 2, 3])
    widthAttr = AttributeSpec("width", [1, 2, 3])
    heavyAttr = AttributeSpec("isHeavy", [False, True])
    classAttr = AttributeSpec("needHelp", ["No", "Yes"])
    ex1 = Example({"height":1, "isHeavy":False, "width":1}, "No")
    ex2 = Example({"height":1, "isHeavy":True, "width":3}, "Yes")
    ex3 = Example({"height":1, "isHeavy":True, "width":2}, "Yes")
    ex4 = Example({"height":3, "isHeavy":True, "width":3}, "Yes")
    ex5 = Example({"height":2, "isHeavy":False, "width":2}, "No")
    exList = ExampleList([ex1, ex2, ex3, ex4, ex5])
    
    print("height, heavy, width gains:")
    print(exList.calcGain(heightAttr, classAttr))
    print(exList.calcGain(heavyAttr, classAttr))
    print(exList.calcGain(widthAttr, classAttr))
    
    print("Attribute chosen:")
    print(exList.chooseAttr([heightAttr, heavyAttr, widthAttr], classAttr))
    

###########################################################################
class DTree(object):
    '''
    A DTree is a decision tree.
    
    The constructor (__init__) just initializes the fields to None.
    To actually make a DTree, though, call either makeLeaf or makeDTree instead.
    (Both of these call the constructor and then set certain fields.)
    
    A DTree has three fields:
    1) question is the AttributeSpec asked by the root of this tree
    2) children is a dictionary mapping question.vals to DTree objects
    3) cls is something in classAttrSpec.__vals - it's one of the valid class values

    One of the following must be true:
    - The tree is just a leaf. So cls has the classification, while children and question are None.
    - The tree is not a leaf. So children and question are set, while cls is None.
    '''

    def __init__(self):
        '''
        Just "declares" the fields and initializes them to None. They get set
        in one of the class methods.
        '''
        self.question = None # The AttributeSpec asked by the root of this tree
        self.children = None # dictionary mapping question.__vals to DTree objects
        
        self.cls = None # A classification - one of the values in classAttrSpec.__vals

    @classmethod
    def makeLeaf(pythonClass, classification):
        '''
        Note the @classmethod annotation. This is the same as a static method in Java.
        Call it like this:
            DTree.makeLeaf(some value in classAttrSpec.vals)
        When you do, pythonClass will have the value DTree. (This is similar to non-class methods,
        in which self takes on the value of the object you called it on.)
        The single argument you pass to makeLeaf will be stored in the classification
        formal parameter.
        
        The result: we return a DTree that is a leaf, assigning the given classification.
        '''
        t = pythonClass()       # construct a DTree object (calls __init__); fields initialized to None
        # Making a leaf, so self.children and self.question stay None
        t.cls = classification  # store the classification for this leaf
        return t

    @classmethod
    def makeDTree(pythonClass, question, children):
        '''
        Call this method:  DTree.makeDTree(question asked at root, children DTrees)
        to make a DTree that is not a leaf.
        '''
        t = pythonClass() # construct a DTree object (calls __init__); fields initialized to None
        t.question = question # store the question asked at this node
        t.children = children # store the child DTrees. Maps t.question.vals to DTrees.
        # Making an internal node (not a leaf), so self.cls stays None
        return t

    def isLeaf(self):
        return self.children == None

    # TO DO
    def classify(self, example):
        '''
        Given this DTree, determine the classification of the given example.
        '''
        return None

    def __repr__(self):
        return self.__reprHelper("")

    def __reprHelper(self, spacing):
        if self.isLeaf():
            return spacing + str(self.cls) + "\n"
        else:
            result = spacing + "[" + str(self.question.getName()) + "\n"
            
            newSpacing = spacing + "    "
            for ans, child in self.children.items():
                if child.isLeaf():
                    result += newSpacing + str(ans) + " -> " + str(child.__reprHelper(""))
                else:
                    result += newSpacing + str(ans) + " :\n  " + str(child.__reprHelper(newSpacing))
                
            result += spacing + "  ]\n"
            return result

    def __makeSpacing(self, n):
        result = ""
        for i in range(n):
            result += " "
        return result

########################################################################
def demoTree(verbose=True):
    '''
    Shows an example of creating a DTree manually (not by induction)
    and using it to classify examples.
    '''

    # Define the attributes that will be used in this domain.
    attrSpecs = [AttributeSpec("Sunny?", [True, False]),
                 AttributeSpec("Warm?", ["Y", "N"]),
                 AttributeSpec("Age?", ["<20", "20-40", ">40"])]

    # Define the class - the question we're asking of each example.
    classAttrSpec = AttributeSpec("Swim?", ["Y", "M", "N"])
    defaultClass = "M"

    '''
    Build this tree manually:
                      Sunny?
               True            False
             Warm?               ans: N
          Y       N
        ans: Y    ans: N
    '''
    warmLeftLeaf = DTree.makeLeaf('Y') # This leaf is ans: Y
    warmRightLeaf = DTree.makeLeaf('N') # This leaf is ans: N
    
    # First arg: the question asked (Warm)
    # Second arg: a dictionary (note the { key:value, ...} notation) mapping from answer-to-Warm? to a tree (or leaf)
    warmTree = DTree.makeDTree(attrSpecs[1], {'Y':warmLeftLeaf, 'N':warmRightLeaf})
    
    sunnyRight = DTree.makeLeaf('N') # This leaf is ans: N
    tree = DTree.makeDTree(attrSpecs[0], {True : warmTree, False : sunnyRight})
    
    if (verbose):
        print("Root question:", tree.question, end='\n\n')
        print("Root children (dictionary from question answer to tree):", tree.children, sep='\n', end='\n\n')
        print("Root, one of the children:", tree.children.get(True), sep='\n', end='\n\n')
        print("Root, another of the children:", tree.children.get(False), sep='\n', end='\n\n')
        print("Root class:", tree.cls, end='\n\n') # None, because root is not a leaf
        
        print("Returning:")
    return tree

def testClassify():
    tree = demoTree(False) # get the tree built above
    print(tree)

    # Set up the list of examples we want to classify with the tree.
    # Here, the actual classification of the examples is also known,
    # so we can check the tree's performance against the actual
    # classifications.
    examples = ExampleList([Example({"Sunny?" : True,
                                     "Warm?" : "Y",
                                     "Age?" : "<20"},
                                    "Y"),
                            Example({"Sunny?" : True,
                                     "Warm?" : "Y",
                                     "Age?" : "20-40"},
                                    "Y"),
                            Example({"Sunny?" : True,
                                     "Warm?" : "N",
                                     "Age?" : "20-40"},
                                    "N"),
                            Example({"Sunny?" : False,
                                     "Warm?" : "N",
                                     "Age?" : "20-40"},
                                    "N"),
                            Example({"Sunny?" : False,
                                     "Warm?" : "Y",
                                     "Age?" : "20-40"},
                                    "N")
                            ])

    # Compare what the tree says (ans) with the actual classification (actual)
    for exID in range(examples.getNumExamples()):
        ex = examples.getExampleAt(exID)
        print("Ans: " + str(tree.classify(ex)) + "     Actual: " + str(ex.getClass()))

########################################################################

# TO DO
def decisionTreeLearning(examples, attrSpecs, classAttrSpec, defaultClass):
    '''
    Given a set of examples, attribute specifications, a class attribute specification,
    and a default class, induce a decision tree on the examples.
    Returns that tree.
    
    examples - an ExampleList object
    attrSpecs - a list of AttributeSpec instances
    classAttrSpec - an AttributeSpec instance, representing the class
    defaultClass - something in classAttrSpec.__vals
    '''
    return None

########################################################################
def runTest(attrSpecs, classAttrSpec, defaultClass, examples):
    # Induce the decision tree.
    tree = decisionTreeLearning(examples, attrSpecs, classAttrSpec, defaultClass)
    print(tree)

    # Run the decision tree on the training set.
    for exID in range(examples.getNumExamples()):
        ex = examples.getExampleAt(exID)
        print("Ans: " + str(tree.classify(ex)) + "     Actual: " + str(ex.getClass()))

########################################################################
def smallTest():
    '''
    A small example of decision tree induction.
    "Is it a good day to go swimming?"
    '''

    # Define the attribute specifications.
    attrSpecs = [AttributeSpec("Sunny?", [True, False]),
                 AttributeSpec("Warm?", ["Y", "N"]),
                 AttributeSpec("Age?", ["<20", "20-40", ">40"])]

    # Define the class attribute specification.
    classAttrSpec = AttributeSpec("Swim?", ["Y", "M", "N"])
    defaultClass = "M"

    # Define the examples on which to induce the decision tree.
    examples = ExampleList([Example({"Sunny?" : True,
                                     "Warm?" : "Y",
                                     "Age?" : "<20"},
                                    "Y"),
                            Example({"Sunny?" : True,
                                     "Warm?" : "Y",
                                     "Age?" : "20-40"},
                                    "Y"),
                            Example({"Sunny?" : True,
                                     "Warm?" : "N",
                                     "Age?" : "20-40"},
                                    "N")
                            ])
    
    runTest(attrSpecs, classAttrSpec, defaultClass, examples)
    
########################################################################
def xorTest():
    # Define the attribute specifications.
    attrSpecs = [AttributeSpec("A", ['F', 'T']),
                 AttributeSpec("B", ['F', 'T'])]
    classAttrSpec = AttributeSpec("A XOR B?", [0, 1])
    defaultClass = 'Y'

    # Define the examples.
    e1 = Example({'A':'F',
                  'B':'F',},
                 0)
    e2 = Example({'A':'F',
                  'B':'T',},
                 1)
    e3 = Example({'A':'T',
                  'B':'F',},
                 1)
    e4 = Example({'A':'T',
                  'B':'T',},
                 0)
    examples = ExampleList([e1, e2, e3, e4])
    
    runTest(attrSpecs, classAttrSpec, defaultClass, examples)
    
#########################################################################
def rnTest():
    '''
    This is a larger example, from Russell and Norvig chapter 18.
    "Should I wait for a table at this restaurant tonight?"
    '''

    # Define the attribute specifications.
    attrSpecs = [AttributeSpec("alt", ['n', 'y']),
                 AttributeSpec("bar", ['n', 'y']),
                 AttributeSpec("fri", ['n', 'y']),
                 AttributeSpec("hun", ['n', 'y']),
                 AttributeSpec("pat", ['none', 'some', 'full']),
                 AttributeSpec("price", ['$', '$$', '$$$']),
                 AttributeSpec("rain", ['n', 'y']),
                 AttributeSpec("res", ['n', 'y']),
                 AttributeSpec("type", ['french', 'italian', 'thai', 'burger']),
                 AttributeSpec("est", ['b010', 'b1030', 'b3060', 'g60'])]
    classAttrSpec = AttributeSpec("Will wait", [False, True])
    defaultClass = False

    # Define the examples.
    e1 = Example({'alt': 'y',
                  'bar': 'n',
                  'fri': 'n',
                  'hun': 'y',
                  'pat': 'some',
                  'price': '$$$',
                  'rain': 'n',
                  'res': 'y',
                  'type': 'french',
                  'est': 'b010'},
                 True)
                            
    e2 = Example({'alt': 'y',
                  'bar': 'n',
                  'fri': 'n',
                  'hun': 'y',
                  'pat': 'full',
                  'price': '$',
                  'rain': 'n',
                  'res': 'n',
                  'type': 'thai',
                  'est': 'b3060'},
                 False)

    e3 = Example({'alt': 'n',
                  'bar': 'y',
                  'fri': 'n',
                  'hun': 'n',
                  'pat': 'some',
                  'price': '$',
                  'rain': 'n',
                  'res': 'n',
                  'type': 'burger',
                  'est': 'b010'},
                 True)
    
    e4 = Example({'alt': 'y',
                  'bar': 'n',
                  'fri': 'y',
                  'hun': 'y',
                  'pat': 'full',
                  'price': '$',
                  'rain': 'y',
                  'res': 'n',
                  'type': 'thai',
                  'est': 'b1030'},
                 True)
    
    e5 = Example({'alt': 'y',
                  'bar': 'n',
                  'fri': 'y',
                  'hun': 'n',
                  'pat': 'full',
                  'price': '$$$',
                  'rain': 'n',
                  'res': 'y',
                  'type': 'french',
                  'est': 'g60'},
                 False)
    
    e6 = Example({'alt': 'n',
                  'bar': 'y',
                  'fri': 'n',
                  'hun': 'y',
                  'pat': 'some',
                  'price': '$$',
                  'rain': 'y',
                  'res': 'y',
                  'type': 'italian',
                  'est': 'b010'},
                 True)
    
    e7 = Example({'alt': 'n',
                  'bar': 'y',
                  'fri': 'n',
                  'hun': 'n',
                  'pat': 'none',
                  'price': '$',
                  'rain': 'y',
                  'res': 'n',
                  'type': 'burger',
                  'est': 'b010'},
                 False)
    
    e8 = Example({'alt': 'n',
                  'bar': 'n',
                  'fri': 'n',
                  'hun': 'y',
                  'pat': 'some',
                  'price': '$$',
                  'rain': 'y',
                  'res': 'y',
                  'type': 'thai',
                  'est': 'b010'},
                 True)
    
    e9 = Example({'alt': 'n',
                  'bar': 'y',
                  'fri': 'y',
                  'hun': 'n',
                  'pat': 'full',
                  'price': '$',
                  'rain': 'y',
                  'res': 'n',
                  'type': 'burger',
                  'est': 'g60'},
                 False)
    
    e10 = Example({'alt': 'y',
                  'bar': 'y',
                  'fri': 'y',
                  'hun': 'y',
                  'pat': 'full',
                  'price': '$$$',
                  'rain': 'n',
                  'res': 'y',
                  'type': 'italian',
                  'est': 'b1030'},
                 False)
    
    e11 = Example({'alt': 'n',
                  'bar': 'n',
                  'fri': 'n',
                  'hun': 'n',
                  'pat': 'none',
                  'price': '$',
                  'rain': 'n',
                  'res': 'n',
                  'type': 'thai',
                  'est': 'b010'},
                 False)
    
    e12 = Example({'alt': 'y',
                  'bar': 'y',
                  'fri': 'y',
                  'hun': 'y',
                  'pat': 'full',
                  'price': '$',
                  'rain': 'n',
                  'res': 'n',
                  'type': 'burger',
                  'est': 'b3060'},
                 True)

    examples = ExampleList([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12])

    runTest(attrSpecs, classAttrSpec, defaultClass, examples)

#########################################################################

def mTest():
    '''
    This example is from Tom Mitchell's "Machine Learning" text.
    "Is today a good day to play tennis?"
    '''

    # Define the attribute specifications.
    attrSpecs = [AttributeSpec("Outlook", ['Sunny', 'Overcast', 'Rain']),
                 AttributeSpec("Temperature", ['Hot', 'Mild', 'Cool']),
                 AttributeSpec("Humidity", ['High', 'Normal']),
                 AttributeSpec("Wind", ['Strong', 'Weak'])]
    classAttrSpec = AttributeSpec("Play tennis?", ['N', 'Y'])
    defaultClass = 'Y'

    # Define the examples.
    e1 = Example({'Outlook': 'Sunny',
                  'Temperature': 'Hot',
                  'Humidity': 'High',
                  'Wind': 'Weak'},
                 'N')
    
    e2 = Example({'Outlook': 'Sunny',
                  'Temperature': 'Hot',
                  'Humidity': 'High',
                  'Wind': 'Strong'},
                 'N')

    e3 = Example({'Outlook': 'Overcast',
                  'Temperature': 'Hot',
                  'Humidity': 'High',
                  'Wind': 'Weak'},
                 'Y')

    e4 = Example({'Outlook': 'Rain',
                  'Temperature': 'Mild',
                  'Humidity': 'High',
                  'Wind': 'Weak'},
                 'Y')

    e5 = Example({'Outlook': 'Rain',
                  'Temperature': 'Cool',
                  'Humidity': 'Normal',
                  'Wind': 'Weak'},
                 'Y')

    e6 = Example({'Outlook': 'Rain',
                  'Temperature': 'Cool',
                  'Humidity': 'Normal',
                  'Wind': 'Strong'},
                 'N')

    e7 = Example({'Outlook': 'Overcast',
                  'Temperature': 'Cool',
                  'Humidity': 'Normal',
                  'Wind': 'Strong'},
                 'Y')

    e8 = Example({'Outlook': 'Sunny',
                  'Temperature': 'Mild',
                  'Humidity': 'High',
                  'Wind': 'Weak'},
                 'N')

    e9 = Example({'Outlook': 'Sunny',
                  'Temperature': 'Cool',
                  'Humidity': 'Normal',
                  'Wind': 'Weak'},
                 'Y')

    e10 = Example({'Outlook': 'Rain',
                  'Temperature': 'Mild',
                  'Humidity': 'Normal',
                  'Wind': 'Weak'},
                 'Y')

    e11 = Example({'Outlook': 'Sunny',
                  'Temperature': 'Mild',
                  'Humidity': 'Normal',
                  'Wind': 'Strong'},
                 'Y')

    e12 = Example({'Outlook': 'Overcast',
                  'Temperature': 'Mild',
                  'Humidity': 'High',
                  'Wind': 'Strong'},
                 'Y')

    e13 = Example({'Outlook': 'Overcast',
                  'Temperature': 'Hot',
                  'Humidity': 'Normal',
                  'Wind': 'Weak'},
                 'Y')

    e14 = Example({'Outlook': 'Rain',
                  'Temperature': 'Mild',
                  'Humidity': 'High',
                  'Wind': 'Strong'},
                 'N')
    
    examples = ExampleList([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14])

    runTest(attrSpecs, classAttrSpec, defaultClass, examples)

