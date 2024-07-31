#set document(
	title: [The Design and Architecture Choice of the Blacklight Library],
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
	Blacklight is python library meant to aid in the creation of topologically optimized DNNs#footnote("Deep Neural Networks") using genetic algorithms. A Blacklight model is intialized by 
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
	#lorem(500)
]]
#paragraph[
	#lorem(100)
]
// The configuration of box and paragraph combos will need be be intelligently decided
// TODO: work out a system that gets rid of this problem
// DO AFTER turning in and presenting project
#box[= Genes
#paragraph[
	#lorem(400)
]]

#box[= Chromosomes
#paragraph[
	#lorem(500)
]]

#box[= Individuals and Population
#paragraph[
	#lorem(500)
]]
