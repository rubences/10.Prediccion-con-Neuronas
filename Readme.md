# Caso práctico para comprender el perceptrón
Ahora vamos a detallar de manera numérica un ejemplo que nos permitirá comprender bien los conceptos que hemos visto hasta el momento. Pero no se asuste, los cálculos no serán complicados y seguirán siendo accesibles en términos de comprensión.

Como hemos visto al principio de este capítulo, el objetivo del perceptrón es clasificar las observaciones. Nosotros le proponemos crear un modelo capaz de determinar si un estudiante, según criterios precisos, puede ser admitido en una universidad de prestigio: IA Academy.

La admisión en esta universidad depende de superar algunos exámenes de entrada. La tabla que aparece a continuación reagrupa los distintos casos de admisiones y rechazos en función de la superación de los exámenes de matemáticas e informática.
Superado el examen de matemáticas	Superado el examen de informática	

Admitido

SÍ	NO	NO
SÍ	SÍ	SÍ
NO	SÍ	NO
NO	NO	NO

Seguro que se ha dado cuenta de que la admisión en la universidad responde a la función lógica AND.
