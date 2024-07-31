// TODO: CHANGE THEME, MAKE FIGURES

#set document(
	title: [The Design and Architecture of the Blacklight Library],
	author: "Jackson Collins",
	keywords: ("blacklight", "machine learning", "dnn", "topology", "python", "neural networks"),
	date: auto
)

#import "@preview/polylux:0.3.1": *

#import themes.simple: *

#set text(font: "Inria Sans")

#show: simple-theme.with(
  footer: [Jackson Collins],
)

#title-slide[
	= The Architecture and Design of the Blacklight Library
	\
	Jackson Collins

	#datetime.today().display()
]

#slide[
	== How Neural Networks Work

	#lorem(20)
]

#focus-slide[
	The Human Touch
]

#slide[
	== Genetic Algorithms

	#lorem(20)
]


#slide[
	== NEAT
]

#focus-slide[
	The Problem with NEAT
]

#slide[
	== Blacklight
	#lorem(20)
]
