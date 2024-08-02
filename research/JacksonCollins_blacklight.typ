#set document(
	title: [The Design and Architecture of the Blacklight Library],
	author: "Jackson Collins",
	keywords: ("blacklight", "machine learning", "dnn", "topology", "python", "neural networks"),
	date: auto
)

#set page(header: context {
	if counter(page).get().first() > 1 [
	#text(size: 10pt)[Jackson Collins#h(1fr)The Design and Architecture of the Blacklight Library]
	]
}, numbering: "1", number-align: bottom + right)

#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge

#set text(font: "Nimbus Roman", lang: "en", size: 12pt)

// #let paragraph(body) = [
// 	// #box(width: 99%, clip: true)[#move(dx: 1%, body)]
// ]

#let paragraph(body) = [
	// pad(y: 1pt, box(width: 99%)[#move(dx: 1%, body)])
	// #pad(y: 1pt, block(width: 99%)[#move(dx: 1%, body)])
	// #pad(y: 1pt, box(width: 99%)[#move(dx: 1%, body)])
	// #set pad(y: 1pt)
	// #block(width: 99%, move(dx: 1%, body))
	#move(dx: 1%, body)#h(1%)
]

#show <para>: {move.with(dx:1%)}

#datetime.today().display()
#line(length: 100%)
#[
	#set heading(outlined: true)
	#align(center)[
		= The Design and Architecture of the Blacklight Library
	]
	#linebreak()
	Jackson Collins#h(1fr)jacksoncofficial3\@gmail.com
]
#line(length: 100%)

=== Abstract
#[
	#box(width: 90%)[
	#set text(size: 11pt)
	#move(dx: 5%)[
		The design and architecture decisions of a project have just as much influence
		over the result as the code within.
	]
]
]

#box[= Introduction
#paragraph[
	Blacklight is a python library meant to aid in the creation of topologically optimized DNNs#footnote("Deep Neural Networks") using genetic algorithms. A Blacklight model is intialized by 
	defining model options through the ModelConfig constructor:
	```python
	config = blacklight.ModelConfig(learning_rate=0.001, ...)
	model = blacklight.create_model(config)
	```
	When the model is defined the `ModelConfig` is coded into _genes_; each gene represents
	one layer in the model. These genes are then collected into chromosomes, where each
	chromosome is a list of genes (```python list[Gene]```). Chromosomes represent individual model
	topologies. The functionality of a chromosome is abstracted into an individual; the individual
	has the ability to mate, therefor it is placed into the population to interact with other 
	topologies.
]]
#[
	#let gene = {
		rect()[Gene \
			#text(size: 10pt)[layer structure \ gene_crossover() \ mutate()]
		]
	}
	#let chromosome = {
		rect()[Chromosome \
			#text(size: 10pt)[model structure \ crossover() \ 
			#footnote("While A implements the logic for some function f, B applies f to itself more generally")<inherit_abstract>mutate()]
		]
	}
	#let individual = {
		rect()[Individual \
			#text(size: 10pt)[
				model object \ 
				mate() \ 
				#footnote(<inherit_abstract>)crossover() \
				get_fitness()
			]
		]
	}
	#let population = {
		rect()[Population \
			#text(size: 10pt)[model pool \
			simulate()
			]
		]
	}
	#[
	#set align(center)
	// #diagram(node((0,0), "A"), node((1,0), "B"), edge((0,0),(1,0), [#text(size: 9pt)[_inherits_]], "->")) \
	#text(size: 9pt)[class B inherits A $->$ ```python class B(A)```] \
	#diagram(spacing: 2cm, {
		let (G, C, I, P) = ((0,0), (1,0), (0,1), (1,1))
		node(G, gene)
		node(C, chromosome)
		node(I, individual)
		node(P, population)
		edge(G,C, "->")
		edge(C,I, "->")
		edge(I,P, "->")
	})]
]
#paragraph[
	The genetic metaphor for designing a neural network allows for complex operations
	such as tuning layer parameters during runtime to be abstracted into simple steps that are 
	intuative for both the library maintainer and the user.
]

// The heading needs to be connected to the first section of the paragraph
// to ensure they don't get split up. This method has the side effect that
// when the paragraph gets too long it will suddenly jump a page.
// to avoid this problem, only bind the heading to the first few sentences
// and then create a new paragraph
#box[= Inspiration
#paragraph[
	The structure and functionality of Blacklight is inspired by a paper called
	\"Evolving Neural Networks through Augmenting Topologies,\" commonly NEAT.
	What NEAT set out to do was formulate a strong method for generating neural networks
	by representing the network as genes and then evolving them. NEAT is state of the art
	for generating neural networks with genetic algorithms but it does have a few shortcomings.
	The implementation of NEAT described in the 2002 paper @stanley:ec02 assumes the network
	to be made entirely out of dense/linear layers, or for the network to be a Multilayer
	Perceptron @enwiki:1236853921. The other primary concern with NEAT is that of computational
	performance in that the model has to be constantly evaluated during evolution. While methods
	for dealing with the second problem are not concrete, Blacklight does intend to at least solve
	the first problem. By allowing the library user to specify certain attributes about the model
	that are then coded into _DNA_, the Blacklight library can evolve more complex networks
	that are able to handle more complex task than a simple MLP#footnote("Multilayer Perceptron").
]]

// The configuration of box and paragraph combos will need be be intelligently decided
// TODO: work out a system that gets rid of this problem
// DO AFTER turning in and presenting project
= Genes
#paragraph[
	When a user defines a models parameters they can be thought of as DNA.
	This DNA tells the library certain things about the model it will be creating,
	such as problem type, minimum/maximum number of layers (and the layer types), and the 
	minimum/maximum number of neurons per layer. The model's DNA is computed by the library to
	formulate genes that define individual layers within the network. During the construction of
	the library it was decided that it would be best to abstract away certain functionalities of
	the chromosome into a gene class. This abstraction allows for certain things about the gene
	to be guaranteed, such as type and size.
	```python
	class Gene:
		def __init__(self, gene_type: object=None, dna: list[Any] | None=None):
			...
	```
	When `gene_type` is not passed in, a random gene is initialized according to the DNA,
	if there is not DNA, the gene is completely random.
]

= Chromosomes
#paragraph[
	Chromosomes represent individual model topologies. `Chromosome` inherits the `Gene` class
	$"Gene" -> "Chromosome"$. Since certain aspects of the gene are confirmed by their existence,
	the `Chromosome` class can use the functionalility implemented in the `Gene` without having
	to worry about edge cases.
	```python
	class Chromosome(Gene)
	```
]

#box[= Individuals and Population
#paragraph[
	The `Individual` and `Population` classes work hand in hand for evaluating and evolving
	the neural network. While a human has 46 chromosomes, a Blacklight model only has 1. The
	`Individual` really exists to abstract some of the functionality of `Chromosome` into
	a way that is easier to deal with. Having the individual also helps keep up the metaphor
	that this project stands on top of. At the end of the the model's evolution it will represent
	one single individual.
	
		#align(center)[$"Chromosome" -> "Individual" -> "Population"$]

	The `Population` is just as it seems, a population of `Individual` being evaluated and bred
	with each other. The `Population` class handles certain aspects of the Blacklight library such
	as running the simulation and defining global rules for the models to follow, for example you
	may want the individuals to mutate a bit before mating, so you pause mating for a few cycles.
]]

= Conclusion
#paragraph[
	The design and architecture of a program heaviliy influences how the program performs, likewise
	the intended functionality of a program will heaviliy influence its design. By representing
	programs in terms of metaphors for what they might be analogous to, complex functionality can
	be abstracted away into simple parts that make up the whole. Simulating a whole individual is 
	complicated, but simulating each of that individual's parts is simple.
]

#pagebreak()
#box[
	#bibliography("./ref.bib", full:true)
]
