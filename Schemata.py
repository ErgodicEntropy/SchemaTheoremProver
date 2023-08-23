#Genetic algorithms, Schema theory, Holland Schema theorem, Building Block hypothesis and Price equation




import math as mt
import random as rd
import numpy as np
import scipy as sc
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from nxpd import draw

Max_Gen = 20

def Fitness(String): #Example fitness of a binary string
    Sum = np.sum(String)
    return Sum

# Schema theory

##Def1: Schemas or Schemata are templates of strings ie; subset of strings that are similar at specific string postion outlined by a placeholder (wildcard symbol)
##Def2: Schemas are platonic abstraction, clustering or categorization of strings based on string positions
##Def3: Schemas are the basis of product topology on strings (Cylinder Sets)
##Def4: Schemas are sequences of sigma* ie; alphabets (0 or 1) with an extra * wildcard symbol (Sigma*L are all the words of length L over the alphabet sigma*)

###Examples: H = [0,1,*,0,0,1,0,1] = [0,1,0,0,0,1,0,1] or [0,1,1,0,0,1,0,1] (Card(H) = 2)

###Schema terminologies:

##Wildcard symbol (Placeholder): placeholders act as arbitrariness-engine ie; they produce variations in order to enumerate concrete strings from a platonic schema (concrete)
##Fixed position: positions that contains the alphabet value ie; positions that are not placeholders => All strings expanded from a schema have the same value at the fixed position of the schema (that's what "similarity" is about) (platonic-abstract)

def String(L): #sigmaL = words of length L defined over alphabet 0,1
    STR = []
    for k in range(L):
        alphabet = int(rd.uniform(0,2))
        STR.append(alphabet)
    return STR

def Empty_Schema(L):
    ES = []
    for k in range(L):
        ES.append('*')
    return ES

def Schema(indexlist,string): #Aposteriori-Inductive method (specific to general)
    H = string
    for k in indexlist:
        H[k] = '*'
    return H

def Schemata(indexlist,L): #Apriori-Deductive method (general to specific)
    H = np.zeros(L)
    for k in indexlist:
        H[k] = '*'
    for j in range(L):
        if j in indexlist:
            continue
        else:
            H[j] = int(rd.uniform(0,2))
            
    return H


def Placeholder_List(Schema):
    placeholder = []
    L = len(Schema)
    for k in range(L):
        if Schema[k] == '*':
            placeholder.append(k)
    return placeholder
            
def Schema_String(Schema): #Greedy Generation algorithm (Recurisve)
    subset = [] #subset of strings => card(subset) = 2*(L-O)
    S = Schema
    placeholder = Placeholder_List(S)
    if len(placeholder) == 0:
        pass
    elif len(placeholder) == 1:
        j = placeholder[0]
        P1 = S.copy()
        P2 = S.copy()
        P1[j] = 1
        P2[j] = 0
        subset.append(P1)
        subset.append(P2)
    else:    
        j = placeholder[0]
        placeholder.pop(0)
        P1 = S.copy()
        P2 = S.copy()
        P1[j] = 1     
        P2[j] = 0
        subset  = subset + Schema_String(P1) + Schema_String(P2)
    
 
    return subset
        
    
        
        
# Str = String(3)
# H1 = Schema([0,1],Str)
# Sub = Schema_String(H1)

# print(H1)

# print(Sub)
    
    
### Population of Strings

N = 20
L = 5

    
    
pm = 0.002 #pm is almost always assumed to be << 1
pc = 0.4
    


## Properties

def Defining_Length(Schema): #last wildcard position - first wildcard position
    placeholder = [] #indexlist or wildcard symbol list
    L = len(Schema)
    for k in range(L):
        if Schema[k] == '*':
            placeholder.append(k)
        else:
            continue
    M = max(placeholder)
    m = min(placeholder)
    return M - m
    

def Order(Schema): #number of fixed/specific positions ie; positions that are not placeholders
    order = 0
    L = len(Schema)
    for k in range(L):
        if Schema[k] == '*':
            order += 1
        else:
            pass
    return order

def Length(Schema): #number of nodes in the program matching the schema or number of nodes of the schema
    O = Order(Schema)
    L = len(Schema)
    return L - O #number of placeholders/wildcard symbols

def SFitness(Schema,Population):
    N = len(Population)
    SI = TI_SchemaMatcher(Population,Schema)
    StrFit = []
    for k in SI:
        StrFit.append(Fitness(Population[k]))
    AvgF = np.sum(StrFit)/N
    return AvgF

def Robustness(Schema):#the measure of arbitrariness in terms of placeholders (quantity and spread) acting as a resistant agaisnt mutation and crossover perturbations
    O = Order(Schema) #measure of the quantity of information or specificity
    DL = Defining_Length(Schema) #measure of how spread the specificity or information is
    dispersion = O * DL
    return 1/dispersion 
    
 
 

def TI_SchemaMatcher(Population,H): #This exctracts the indices of the strings belonging/matching with H out of the population
    N = len(Population)
    StringIndex = []
    for k in range(N):
        String = Population[k]
        if Matching(H,String) == True:
            StringIndex.append(k)
    return StringIndex


   
def TI_Matching_Number(Population,H): #Number of strings that belong or match the schema H 
    N = len(Population)
    StringCluster = []
    for k in range(N):
        String = Population[k]
        if Matching(H,String) == True:
            StringCluster.append(String)
    
    return len(StringCluster)
      
StrPop = []
for k in range(N):
    Str = String(L)
    StrPop.append(Str)
    
LOLN = 10   #Monte Carlo limit
MetaPop = [StrPop]
n = 1
while n < LOLN:
    SamplePop = []
    for k in range(N):
        Strr = String(L)
        SamplePop.append(Strr)
    MetaPop.append(SamplePop)
    
# Building Block hypothesis: Heuristic Engineering
# Genetic Algorithm (binary strings): Heuristic example
            
def Population_t(t): #Rules are dictated by the heuristic (Selection -> Crossover -> Mutation)
    if t == 0:
        return StrPop
    else:
        StrPopt1 = Population_t(t-1)
        K = len(StrPopt1)
        F = []
        for k in range(K):
            f = Fitness(StrPopt1[k])
            F.append(f)
        #Selection
        Selection = []
        for j in range(K):
            sp = F[j]/np.sum(F)
            Selection.append(sp)
        SelectedPop = StrPopt1.copy()
        pth = rd.uniform(0,1)
        for l in range(K):
            if Selection[l] > pth:
                SelectedPop.append(StrPopt1[l])
            else:
                pass
        #Crossover
        CrossoverPop = SelectedPop.copy()
        slen = len(CrossoverPop)
        mid = L//2
        
        for j in range(slen):
            pcth = rd.uniform(0,1)
            if pc > pcth:
                r = int(rd.uniform(0,slen))
                u = int(rd.uniform(0,slen))
                while r == u:
                    r = int(rd.uniform(0,slen))
                    u = int(rd.uniform(0,slen))
                X = CrossoverPop[r][:mid]
                CrossoverPop[r][:mid] = CrossoverPop[u][:mid]
                CrossoverPop[u][:mid] = X
        
        #Mutation
        MutatedPop = CrossoverPop.copy()
        for k in range(slen):
            pmth = rd.uniform(0,1)
            if pm >= pmth:
                u = int(rd.uniform(0,L))
                MutatedPop[k][u] = int(not bool(MutatedPop[k][u] == 1))
                
        Populationt = MutatedPop.copy()
        return Populationt
    
MetaPopulation = [StrPop]
for j in range(1,Max_Gen):
    IPOP = Population_t(j) #Intermediary population
    MetaPopulation.append(IPOP)
        
def SchemaMatcher(H,t):
    Pop = MetaPopulation[t]
    SIT = TI_SchemaMatcher(Pop,H)
    return SIT               
                     
def Matching_Number(H,t): #Number of strings that belong or match the schema H at generation t
    Pop = MetaPopulation[t]
    sc = TI_Matching_Number(Pop,H)
    return sc

def RecursiveMatchingNumber(H,t):
    if t == 0:
        return TI_Matching_Number(StrPop,H)
    else:
        Population = MetaPopulation[t]
        m = Matching_Number(H,t-1)
        f = SFitness(H,Population)
        F = []
        SCP = SchemaCluster(Population)
        C = len(SCP)
        for k in range(C):
            F.append(SFitness(SCP[k],Population))
        AvgF = np.mean(F)
        return m*f/AvgF
    
def Disruption(Schema,t): #A schema is said to be disrupted if the parent strings that match produce children that don't. As long as we are examining the first-generation children only, dicussions about atavism are ignored as we only care about the intergenerational interval [t,t+1]
    th = 0.5 #logistic threshold
    Popt1 = MetaPopulation[t]
    Popt2 = MetaPopulation[t+1]
    MN1 = Matching_Number(Schema,t)
    MN2 = Matching_Number(Schema,t+1)
    N1 = len(Popt1)
    N2 = len(Popt2)
    ratio1 = MN1/N1
    ratio2 = MN2/N2
    Is_Disrupted = bool(ratio1/ratio2 > th)
    return Is_Disrupted
    


def Propagation(Schema,t): #Atavism as a special case
    Is_Disrupted = Disruption(Schema,t)
    Propagated = not Is_Disrupted
    return Propagated

def Atavism(Schema):
    Is_Disrupted = Disruption(Schema,0)
    for j in range(1,Max_Gen):
        Is_Disrupted = Is_Disrupted * Disruption(Schema,j)
    Atavism  = not Is_Disrupted
    return Atavism

def ProbDisruption(Schema):
    O = Order(Schema)
    DL = Defining_Length(Schema)
    l = Length(Schema)  #l is the length of the code
    pdm = O*pm
    pdc = (DL/l+1)*pc
    return pdm + pdc
    

def ProbSurvival(Schema):
    pd = ProbDisruption(Schema)
    ps = 1 - pd
    return ps



## Operators

def Matching(Schema, String):
    Match = True
    L = len(Schema)
    placeholder = Placeholder_List(Schema)
    for j in range(L):
        if j in placeholder:
            continue
        else:
            Match = Match * bool(Schema[j] == String[j])
    return Match
    
    
def Expansion(Schema): #Expansion is a mapping from schemas to strings/words
    StringSet = Schema_String(Schema)
    return StringSet

def Total_Compression(wordset): #Compression is a mapping from strings/words to schemas (Compression requires at least 2 strings/words as input since ''similarity'' is a binary relation)
    Schema = []
    W = len(wordset)
    sampleword = wordset[0]
    L = len(sampleword)
    j = 0
    while j < L:
        similarj = True
        for k in range(1,W):
            similarj = similarj * bool(sampleword[j] == wordset[k][j])
        if similarj == True:
            Schema.append(sampleword[j])
        else:
            Schema.append('*')
        
        j = j + 1
        
    return Schema

# Sc = Total_Compression([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])

# print(Sc)

def Binary_Compression(Word1,Word2):
    Binary_Schema = []
    L = len(Word1)
    for k in range(L):
        if Word1[k] == Word2[k]:
            Binary_Schema.append(Word1[k])
        else:
            Binary_Schema.append('*')
            
    return Binary_Schema

### Observations about compression:

## Compression Ramsey-Percolation effect: Compression leads to different results depending on the number of strings considered
## Length = Size - Order, increases with the number of distinct strings considered in compression. And given that min(Order) = 1, then the max(Length) = Size - 1 
## Empty Schema cannot be reached by compression, since that would give a definitional contradiction because Order(Empty Schema) = 0 therefore it cannot define any subset of strings (permutation buffer [principle of indifference, maximum entropy]) and thus it's without functionality
## Expansion of Empty Schema is a permutation of alphabets
## Dollo law-Hysteresis effect of Binary Compression is due to the non-absorptive (non-reductionist) common-input and emergent nature of binary compression-abstraction. This problem is solved by Schematic Lattice which is exhaustive.
## Properties: BC(a,b) = BC(c,d) <=> (a = c and b = d) or (a = d and b = c), BC is symmetric, non-transitive and non-reflexive, BC(a,b) != BC(c,b) if a != c

# The compression and expansion operators form a Galois connection, where ↓H  is the lower adjoint and ↑H  the upper adjoint

def SchemaCluster(Population): #This extracts schemas from the population of strings (number of schemas and schemas themselves depend on the compression parameters)
    PopLen = len(Population)
    R = int(rd.uniform(0,2))
    if R == 1:
        return Total_Compression(Population) #This case is unique both in terms of number (1) and schema value
    else:
        MetaSchema = []
        MaxIter = PopLen//4 #number of schemas to be obtained. This number satisfies certain constraints as there are some values that are forbidden
        t = 0 
        while t < MaxIter:
            r = int(rd.uniform(0,PopLen)) 
            u = int(rd.uniform(0,PopLen)) 
            while r == u:
                r = int(rd.uniform(0,PopLen)) 
                u = int(rd.uniform(0,PopLen))
                
            BC = Binary_Compression(Population[u],Population[r])
            while BC in MetaSchema:
                r = int(rd.uniform(0,PopLen)) 
                u = int(rd.uniform(0,PopLen)) 
                while r == u:
                    r = int(rd.uniform(0,PopLen)) 
                    u = int(rd.uniform(0,PopLen))
                BC = Binary_Compression(Population[u],Population[r])
                
            MetaSchema.append(BC)
            
            t = t + 1
            
    return MetaSchema
         

        
        


def Partial_Order(Schema1,Schema2): #Partial order in the sense of set inclusion ( or cardinality) = reflexive, antisymmetric and transitive => (SchemaSet,⊆) is a poset
    Exp1 = Expansion(Schema1)
    Exp2 = Expansion(Schema2)
    Card1 = len(Exp1)
    Card2 = len(Exp2)
    PO = bool(Card1 > Card2) #or subset inclusion
    return PO
    

def SchematicCompletion(WordsSet): #WordsSet = Population = A
    SC = WordsSet.copy()
    L = len(WordsSet)
    j = 0
    while j < L-1:
        for k in range(j+1,L):
            BS = Binary_Compression(WordsSet[j], WordsSet[k])
            if BS in SC:
               pass
            else:
                SC.append(BS)
        j = j + 1
    SchematicCompletion = SC.copy()
    e = Empty_Schema(L)
    Universal_Schema = Total_Compression(WordsSet)
    if e in SchematicCompletion:
        pass
    else:
        SchematicCompletion.append(e)
    if Universal_Schema in SchematicCompletion:
        pass
    else:
        SchematicCompletion.append(Universal_Schema)
    return SchematicCompletion
        


# S = SchematicCompletion(StrPop)
# print(S)    
    

def Levenshtein_Star(Word1,Word2):
    star1 = Word1.count('*')
    star2 = Word2.count('*')
    return abs(star2-star1)
    
    
#The point of Schematic Completion is compression exhaustion by combining Total Compression and brute-forcing all possible Binary Compressions

def SchematicLattice(WordsSet): #Complete Lattice (Hasse diagram): Complete lattice is a poset with join (supremum or least upper bound) and meet (infimum or greatest lower bound)
    L = len(WordsSet)
    SL = len(WordsSet[0]) 
    ScComp = SchematicCompletion(WordsSet) #first L elements are that of WordsSet, L+1 -> CL-2 are schemas, CL-1 is the empty schema
    CL = len(ScComp)
    # emptyschema = ScComp[CL-1] #HasseDiagram[CL-1]
    HasseDiagram = np.zeros((CL,CL))
    for k in range(L):
        HasseDiagram[CL-1][k] = 1
        HasseDiagram[k][CL-1] = 1
    
    Similarity = np.zeros((CL-1,CL-1))
    for k in range(CL-1): #Words/Strings + Schemas
        for j in range(L+1,CL-1): #Only Schemas
            if k == j: #only happen when comparing schemas to schemas ie; when k is in [L+1,CL-1]
                pass
            else:
                B = True
                for m in range(SL):
                    if ScComp[k][m] == '*' or ScComp[j][m] == '*':
                        pass
                    else:
                        B = B * bool(ScComp[k][m] == ScComp[j][m])
                Similarity[k][j] = B
                Similarity[j][k] = B 
                                  
    for j in range(CL-1): #Words/Strings + Schemas
        for k in range(L+1,CL-1): #Schemas
            if Levenshtein_Star(ScComp[k],ScComp[j]) == 1 and Similarity[k][j] == True:
                HasseDiagram[k][j] = 1
                HasseDiagram[j][k] = 1
            else:
                pass
    return HasseDiagram

# HASSE = SchematicLattice(StrPop)
# print(HASSE)
    
#Hasse Diagram of the Schematic Lattice:
##1- All words/strings are connected to the empty schema
##2- set of words/strings and set of schemas of a fixed length are the stable sets of a bipartite graph ie; no to two words/strings are connected jsut as no two schemas of the same length are conntected either.
 
def HasseDiagram(WordsSet):
    HD = SchematicLattice(WordsSet)
    lhd = len(HD)
    StringLabels = []
    t = 0
    while t < lhd:
        samplestring = HD[t]
        label = ''.join(str(k) for k in samplestring)
        StringLabels.append(label)
            
    HasseGraph = nx.Graph()
    for m in range(lhd):
        HasseGraph.add_node(m,label=StringLabels[m])
    for k in range(lhd):
        for j in range(lhd):
            if HD[k][j] == 1 or HD[j][k] == 1:
                HasseGraph.add_edge(k,j)
    
    nx.draw(HasseGraph, with_labels = True, font_weight = 'bold')
    plt.show()


# HasseDiagram(StrPop)
    



# Holland Schema theorem: Computer-assisted Verification/Proof

def SchemaTheoremProver(t):
    Pop = MetaPopulation[t]
    SCluster = SchemaCluster(Pop)
    H = SCluster[int(rd.uniform(0,len(SCluster)))] #Random Schema 
    Mt1 = Matching_Number(H,t)
    Mt2 = RecursiveMatchingNumber(H,t)
    psu = ProbSurvival(H)
    return bool(Mt1 >= Mt2*psu)
    
    
BooleanTest = []
for T in range(Max_Gen):
    B = int(SchemaTheoremProver(T))
    BooleanTest.append(B)
 
positive = np.sum(BooleanTest)   
Proof = bool(positive > Max_Gen//2)

print(Proof)





## Hypotheses:
###H1: Probability(Atavism) is non-zero but small therefore neglected  (Atavism is when a string matching schema H appears from scratc either via mutation or crossover of children from next generation whose parents didn't match schema H)
###H2: Genetic algorithms maintain or manipulate infinite population and never converge to a finite-size population.


## Limitations:
###L1: Sampling error -> Premature convergence towards solutions with no selective advantage (common in multimodal optimization)
###L2: HST cannot explain the effectiveness of GAs since it doesn't distinguish between problems in which GA performs poorly from problems in which GA performs efficiently.






##Applications: Cognitive Benchmarking, Cognitive Constructivism (Schemata), Comprehensive Input hypothesis (Semantic Selection via Tokenization-Encoding)...etc



def Tokenization(data,SL): ##It is a good heuristic that SL >>> L
    Strings = {}
    Schemas = {}
    L = len(data)
    TabuString = []
    for k in range(L):
        Str = String(SL)
        while Str in TabuString:
            Str = String(SL)
        Strings[data[k]] = Str
        TabuString.append(Str)
    return Strings
    
    
def TokenizedFitness(data):
    pass
    #Retrieve feedback
    #Fitness Matcher: Matching Tokenized Fitness with Binary Fitness (Schemas and Strings)
