// TODO: CHANGE THEME, MAKE FIGURES

#set document(
	title: [The Design and Architecture of the Blacklight Library],
	author: "Jackson Collins",
	keywords: ("blacklight", "machine learning", "dnn", "topology", "python", "neural networks"),
	date: auto
)

#import "@preview/polylux:0.3.1": *

#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge

#import "@preview/suiji:0.3.0": *

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
	// #let input_layer = ()
	// #let dense_layer = ()
	// #let output_layer = ()
	// #for y in range(1,6){
	// 	input_layer.push((0,y))
	// }
	// #for x in range(1,3){
	// 	for y in range(7){
	// 		dense_layer.push((x, y))
	// 	}
	// }
	// #for y in range(2,5){
	// 	output_layer.push((3, y))
	// }
	// #diagram(
	// 	spacing: 0.0em,
	// 	for i in range(input_layer.len()){
	// 		node(input_layer.at(i), if i == 2 {[#line(length: 1em, angle: 90deg, stroke: (dash: "dotted"))]}else{[#circle()]})
	// 	},
	// 	for i in range(dense_layer.len()){
	// 		node(dense_layer.at(i), if i == 3 or i == 10{[#line(length: 1em, angle: 90deg, stroke: (dash: "dotted"))]}else{[#circle()]})
	// 	},
	// 	for i in range(output_layer.len()){
	// 		node(output_layer.at(i), circle())
	// 	},
	// 	for i in range(input_layer.len()){
	// 		edge(input_layer.at(i), dense_layer.at(i))
	// 	}
	// )
	#show image: set align(center)
	#figure(
		supplement: none,
		image("./Colored_neural_network.svg", height: 70%),
		caption: [#text(size: 16pt)[Glosser.ca, CC BY-SA 3.0 \<#link("https://creativecommons.org/licenses/by-sa/3.0")\>, via Wikimedia Commons)]],
	)
]

#focus-slide[
	The Human Touch
	#show circle: set align(center)
	#circle(stroke: 2pt)[0.42]
]

#slide[
	== Genetic Algorithms
	#figure(supplement: none,
	image("./TwoPointCrossover.svg", height: 70%),
	caption: [#text(size: 16pt)[R0oland, CC BY-SA 3.0 \<#link("https://creativecommons.org/licenses/by-sa/3.0")\>, via Wikimedia Commons]])
]


#focus-slide[
	That's NEAT
]

#slide[
	#set align(center)
	#v(40%)
	#set text(size: 100pt)
	// #set text(font: "Noto Color Emoji")
	#image("./fire.svg")
]

#slide[
	== Blacklight
	#figure(supplement: none,
	image("./Chromosomal_Crossover.svg", height: 70%),
	caption: [#text(size: 16pt)[Abbyprovenzano, CC BY-SA 3.0 \<#link("https://creativecommons.org/licenses/by-sa/3.0")\>, via Wikimedia Commons]])
]

#slide[
	#set text(size: 16pt)
	#bibliography("./ref.bib", full: true)
]
