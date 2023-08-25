# SchemaTheoremProver
This project is about testing Holland Schema theorem statistically against the randomness of genes, alleles and chromosomes via the law of large numbers. 

# Schema Theory:
Schema theory is the study of schemas or schematas from the point of view of computer science and discrete mathematics. Schema theory has a lot of applications ranging from Natural Language Processing to Computational Neuroscience (neural encoding...etc) to Bioinformatics (Genome Assembly, Genome Segmentation, Optimal Alignment...etc) to Cognitive Benchmarking, Cognitive Constructivism (Schemata), Comprehensive Input hypothesis (Semantic Selection via Tokenization-Encoding)...etc. Schemas or Schemata are usually dicussed within the realm of constructivist learning theories such Jean Piaget's cognitive construcitivism. But for our purpose, they are generally the same entity. However, for the sake and scope of this project, we are going to define them in the discrete-mathematical-computer-science sense.

## Definition:
1. Def1: Schemas or Schemata are templates of strings ie; subset of strings that are similar at specific string postion outlined by a placeholder (wildcard symbol)
2. Def2: Schemas are platonic abstraction, clustering or categorization of strings based on string positions
3. Def3: Schemas are the basis of product topology on strings (Cylinder Sets)
4. Def4: Schemas are sequences of sigma* ie; alphabets (0 or 1) with an extra * wildcard symbol (Sigma*L are all the words of length L over the alphabet sigma*)

### Examples: 
H = [0,1,*,0,0,1,0,1] = [0,1,0,0,0,1,0,1] or [0,1,1,0,0,1,0,1] (Card(H) = 2)

### Schema terminologies:

1. Wildcard symbol (Placeholder): placeholders act as arbitrariness-engine ie; they produce variations in order to enumerate concrete strings from a platonic schema (concrete)
2. Fixed position: positions that contains the alphabet value ie; positions that are not placeholders => All strings expanded from a schema have the same value at the fixed position of the schema (that's what "similarity" is about) (platonic-abstract)

## Properties: 
### Defining Length:
Defining Length is defined as the last wildcard position - first wildcard position. It's the measure of spread of the placeholders are

### Order

Order is defined as the number of fixed/specific positions ie; positions that are not placeholders. It's the measure of quantity of information in the binary string/chromosome.

### Fitness

Fitness of a schema is defined as the average fitness of all strings/chromosomes that match this schema

### Length

Length is defined as the number of nodes in the program matching the schema or number of nodes of the schema. But for our context, length is the number of placeholders/wildcard symbols. Therefore, Length = size - order

### Matching number

Matching number of a given schema is the number of strings that match or belong to this schema in a given generation t.

### Propagation vs Disruption 

1. Disruption is the intergenerational information loss. A schema is disrupted when the parent strings that match the schema produce children that don't. This can take place due to 3 causes related to order, defining length and/or fitness of the schema. Meanwhile fitness can be collectively attributed to a given schema based on the nature of density-dependence allele frequency, order and defining length are intrinsic to the schema in the sense of schema robustness ie; how much a given schema is resistant to external pertubators such as intergenerational stochastic mutation and crossover.
2. Propagation is the intergenerational inheritance of a given schema characteristics and form (defined in the platonic sense). A schema is said to be propagated if the there are strings in the current or next generations that match it. These strings do not have to come from parents that match the schema themselves as it could take the form of atavism ie; a sudden formation of a matching strngs from scratch via either stochastic mutation or crossover, this probability is very minimal to the point of neglection however.


## Operators:
### Matching: 
Matching is the abstract-concrete binary relation between a given schema and a string. A string (or chromosome) is said to match or belong to a schema if its genes match that of the schema.

### Expansion
Schema expansion involves relaxing the constraints of a schema, which results in a more general template. When expanding a schema, some of the fixed values or wildcard positions are relaxed, allowing a wider range of strings to match the new schema. This operator is analogous to generalizing a pattern.

For example, consider a schema that represents DNA sequences where certain positions are fixed to particular nucleotides. Through expansion, some of these positions could be turned into wildcards, indicating that any nucleotide is acceptable at those positions. This broadens the applicability of the schema to a larger subset of sequences.
### Compression
Schema compression is the opposite of expansion. It involves making a schema more specific by tightening its constraints. This can be achieved by converting some of the wildcard positions into fixed values. Compression makes the schema match a more restricted subset of strings by specifying more precise patterns.

Using the DNA sequence example, a schema might initially have wildcards at certain positions, allowing any nucleotide. By compressing the schema, these wildcards could be replaced with specific nucleotide values, making the schema applicable to only a subset of sequences that share those exact nucleotides at those positions.

### Completion
Schema completion involves filling in the missing information within a schema to make it applicable to a broader set of strings. This is particularly useful when dealing with partial or incomplete schemas. Completion aims to create a schema that captures the patterns present in a group of strings that may share similarities but have variations in some positions.

Continuing with the DNA sequence example, if a schema is missing values at certain positions, completion could involve analyzing the strings that match the partial schema and inferring the missing nucleotides based on the observed patterns. This process generates a completed schema that better represents the shared characteristics of the strings.

These schema operators interact with each other and with the genetic operators (crossover and mutation) to drive the exploration and exploitation of the solution space in genetic algorithms. They allow for the adaptation and evolution of schemas, leading to the discovery of better solutions over generations.

# Holland Schema theorem
The Holland Schema Theorem is a fundamental result in the theory of genetic algorithms, which are optimization algorithms inspired by natural evolution. The theorem was introduced by John Holland in the 1970s. It provides insights into how genetic algorithms manipulate and evolve populations of solutions to find better solutions over time.

The theorem is based on the concept of "schemas," which are patterns that represent subsets of strings in a population. A schema is defined by a string of symbols where each symbol represents a value at a specific position in the string. Symbols can be actual values, wildcards (denoted by '*'), or both. The fitness of a string is determined by the number of instances of different schemas that match the string. Schemas capture commonalities among strings that contribute to higher fitness.

Holland's Schema Theorem states that, under certain conditions, the average fitness of the population will increase over generations because short, low-order schemas with above-average fitness increase exponentially over generations. These conditions include:

Fitness Determination: The fitness of a string depends only on the instances of schemas in the string.
Crossover and Mutation: The genetic operators (crossover and mutation) maintain the linkage between positions of schemas. In other words, these operators should not break the beneficial combinations of schema patterns that contribute to higher fitness.

Schema theorem is defined by the following inequality: m(H,t + 1) ≥ (1 - c) m(H,t) + c (1 - p)^l m(t) + O(p^l)

Where:

m(H,t) is the average matching number of schema H in the population at generation t.
m(H,t + 1) is the average matching number of schema H in the population at the next generation t + 1.
c is the crossover rate, representing the proportion of individuals generated via crossover.
p is the mutation rate, representing the probability of a single position being mutated.
l is the length of the schema.
O(p^l) represents the order term, accounting for contributions from schemas with mutations.


## Hypotheses:
1. H1: Probability(Atavism) is non-zero but small therefore neglected  (Atavism is when a string matching schema H appears from scratc either via mutation or crossover of children from next generation whose parents didn't match schema H.
2. H2: Genetic algorithms maintain or manipulate infinite population and never converge to a finite-size population.


## Limitations:
1. L1: Sampling error -> Premature convergence towards solutions with no selective advantage (common in multimodal optimization).
2. L2: HST cannot explain the effectiveness of GAs since it doesn't distinguish between problems in which GA performs poorly from problems in which GA performs efficiently.


# Building Block Hypothesis
The Building Block Hypothesis, also proposed by John Holland, complements the Schema Theorem. It posits that the evolution of solutions in genetic algorithms occurs through the recombination of small, low-order schemata known as "building blocks." Building blocks are short sequences of symbols that exhibit a specific pattern or property. They can be thought of as the atomic units of genetic evolution.

The hypothesis suggests that successful evolution happens by combining and preserving these building blocks over generations. Recombination allows the favorable combinations of building blocks to create more fit solutions, and mutation helps explore new combinations.

# Difference between Schema theorem and Building Block hypothesis:
The key difference between the Holland Schema Theorem and the Building Block Hypothesis lies in their focus and scope. The Schema Theorem is a formal result that deals with the mathematical properties of schemas, such as how they contribute to fitness and how they are affected by genetic operators. It provides a theoretical foundation for understanding the evolution of populations.

On the other hand, the Building Block Hypothesis is a more qualitative idea that emphasizes the role of small, fit patterns (building blocks) in the evolutionary process. It provides an intuitive perspective on how genetic algorithms work by highlighting the importance of preserving and recombining advantageous patterns within solutions.

In essence, the Schema Theorem provides the mathematical underpinning, while the Building Block Hypothesis offers a conceptual framework for understanding the practical mechanisms behind genetic algorithm evolution.

