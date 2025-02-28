from langchain_text_splitters import CharacterTextSplitter
text = """
Biology is a fascinating and diverse field of science that explores the living world and its intricacies. It encompasses the study of life, its origins, diversity, structure, function, and interactions at various levels, from molecules and cells to organisms and ecosystems. In this 1000-word essay, we will delve into the core concepts of biology, its history, key areas of study, and its significance in shaping our understanding of the natural world.

The Origin and History of Biology:
The roots of biology can be traced back to ancient civilizations, where observations of plants and animals formed the basis of early biological knowledge. The Greeks, notably Aristotle, made significant contributions to the field, with his comprehensive studies on animals and classification systems. However, it was not until the 17th and 18th centuries that modern biology began to take shape, thanks to pioneers like Carl Linnaeus, who developed the binomial nomenclature, and Antonie van Leeuwenhoek, who made groundbreaking discoveries using the microscope.

Evolutionary Theory and the Impact of Charles Darwin:
One of the most revolutionary ideas in biology is Charles Darwin's theory of evolution by natural selection. In his seminal work, "On the Origin of Species" (1859), Darwin proposed that species evolve over time through a process of adaptation to their environment, leading to the diversity of life we observe today. This theory provided a unifying explanation for the vast array of life forms and their similarities, forever changing the way biologists approached their studies.

Cell Theory and the Foundation of Modern Biology:
In the 19th century, advances in microscopy and cellular studies led to the formulation of the cell theory, another critical milestone in biology. This theory, developed by Matthias Schleiden, Theodor Schwann, and Rudolf Virchow, states that all living organisms are composed of one or more cells, and cells are the fundamental units of life. The cell theory laid the groundwork for understanding the structure and function of living organisms and opened up new avenues of research.

Genetics and the Discovery of DNA:
In the early 20th century, the field of genetics emerged, with Gregor Mendel's work on inheritance and the study of heredity in pea plants. The rediscovery of Mendel's laws paved the way for the science of genetics and our understanding of how traits are passed from one generation to the next. The landmark discovery of the structure of DNA by James Watson and Francis Crick in 1953 revolutionized biology further, revealing the genetic basis of life and leading to the field of molecular biology.

Ecology and the Interconnectedness of Life:
Ecology is a fundamental branch of biology that explores the relationships between organisms and their environment. It examines how living organisms interact with each other and their surroundings, from individual species to entire ecosystems. Understanding ecological principles is crucial for addressing environmental challenges and conserving biodiversity.

Physiology and the Study of Life Processes:
Physiology is the study of the normal functions of living organisms and their parts. It encompasses a broad range of research, from the workings of individual cells to the functioning of whole organ systems in humans and other animals. Understanding physiology is essential for medical advancements and improving human health.

Evolutionary Biology and Adaptation:
Building on Darwin's theory, evolutionary biology explores how organisms change over time, driven by natural selection and other evolutionary mechanisms. It sheds light on the adaptations that allow species to survive and thrive in their environments, as well as the forces that can lead to extinction.

Microbiology and the Invisible World:
Microbiology is the study of microorganisms, such as bacteria, viruses, and fungi. Despite their small size, these organisms have a profound impact on life on Earth. Microbiologists study the roles of microorganisms in disease, food production, environmental processes, and their potential applications in biotechnology.

Biotechnology and its Applications:
Biotechnology is an interdisciplinary field that applies biological principles and techniques to develop products and technologies for various purposes. It has led to significant advancements in medicine, agriculture, and industry. Genetic engineering, gene editing, and the production of biopharmaceuticals are some examples of biotechnological applications.

Conservation Biology and Biodiversity:
Conservation biology is a crucial field that focuses on the preservation of biodiversity and the sustainable use of natural resources. It addresses the threats to species and ecosystems, such as habitat destruction, pollution, and climate change, aiming to protect and restore the delicate balance of nature.

Conclusion:
Biology is a multifaceted and dynamic discipline that continues to evolve and uncover new wonders of the living world. From the earliest observations of nature to the groundbreaking discoveries of genetics and evolution, biology has shaped our understanding of life and its complexity. It plays a vital role in addressing global challenges, from understanding diseases and developing treatments to safeguarding our environment and conserving biodiversity. As technology and research methods continue to advance, the future of biology promises even greater revelations, offering us a deeper appreciation of the magnificent tapestry of life that surrounds us.

"""

# Brak nakładania się fragmentów:
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
chunk_size=50, chunk_overlap=0, separator="\n",
)
texts = text_splitter.split_text(text)
print(f"Liczba fragmentów tekstu bez nakładania się: {len(texts)}")
# Uwzględnij nakładanie się fragmentów:
text_splitter = CharacterTextSplitter.from_tiktoken_encoder( chunk_size=50, chunk_overlap=48, separator="\n",
)
texts = text_splitter.split_text(text)
print(f"Liczba fragmentów z nakładaniem się: {len(texts)}")

