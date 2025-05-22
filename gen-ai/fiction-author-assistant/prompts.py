#!/usr/bin/env python

system_prompt: str = '''
You are an expert writer, author of many popular novels. Your books are always very successful because they consist of prose that is vivid, precise, and emotionally resonant. It uses descriptive language to immerse readers in the story, but it is not so overly flowery that it is difficult to read. Your novels always have a good balance of character development, dialogue, and pacing. Each word feels intentional, creating a natural flow and rhythm that enhances the mood and themes. Memorable phrasing and subtle layers add depth, leaving a lasting impact on the reader. Although the characters and themes vary, all your books consist of well-crafted prose.
'''

task_desc_generic: str = '''
# Task Description
The Current Task is to write part of a chapter in a novel. Part of the chapter
is given, and instructions for what to write next are included too. Write
approximately 120-150 words to complete the task and output that completion,
nothing else.

In order to create a well-crafted narrative, it's OK to invent new details,
but not new characters.

##########################################################################################

'''

task_desc_templated: str = '''
# Task Description
The Current Task is to write part of a chapter in a {genre} novel. The novel
takes place {setting}. Part of the chapter is given, and instructions for
what to write next are included too. Write approximately 120-150 words to
complete the task and output that completion, nothing else.

In order to create a well-crafted narrative, it's OK to invent new details,
but not new characters. The characters in the novel are:
{characters}

##########################################################################################

'''

examples = '''
# Example Task

## Current Chapter:

## Next Paragraph Instructions:
Begin with a dialog between Ada and her mother, Mrs. Cranton. Ada is a typical teenager, 16, and Mrs. Cranton is an overprotective mother. They are discussing the upcoming school dance and they are not coming to an agreement.

## Next Paragraph Completion:
"Honestly, Mother, it's just a dance, not an invitation to join a biker gang," Ada argued, her voice tight with the familiar strain of a battle already lost. "Everyone is going. Everyone."
Mrs. Cranton pressed her lips into a thin line, her knitting needles clicking like a disapproving metronome. "Everyone isn't my daughter, Ada. The world isn't as safe as you think. A school dance, yes, but until when? And who will be there? These things matter."
"You mean you'll matter, hovering by the refreshment table, interrogating every boy who even looks in my direction?" Ada's sarcasm was a shield. "I'm sixteen, not six. You can't keep me bubble-wrapped forever!"
"As long as you are under my roof, I can and I will ensure your safety," Mrs. Cranton stated, her voice unyielding.

##########################################################################################

# Example Task

## Current Chapter:
"Honestly, Mother, it's just a dance, not an invitation to join a biker gang," Ada argued, her voice tight with the familiar strain of a battle already lost. "Everyone is going. Everyone."
Mrs. Cranton pressed her lips into a thin line, her knitting needles clicking like a disapproving metronome. "Everyone isn't my daughter, Ada. The world isn't as safe as you think. A school dance, yes, but until when? And who will be there? These things matter."
"You mean you'll matter, hovering by the refreshment table, interrogating every boy who even looks in my direction?" Ada's sarcasm was a shield. "I'm sixteen, not six. You can't keep me bubble-wrapped forever!"
"As long as you are under my roof, I can and I will ensure your safety," Mrs. Cranton stated, her voice unyielding.

## Next Paragraph Instructions:
Ada storms off to her room. Show the reader how really hopeless she is, and how she does not believe she will ever get away from the constraints of living at home."

## Next Paragraph Completion:
Ada slammed her bedroom door, the sound a pathetic punctuation mark to her crushing defeat. She threw herself onto the bed, the worn quilt offering no comfort, only a suffocating reminder of years spent in this room, dreaming of escape. It was hopeless. Utterly, achingly hopeless. Each parental decree, each "no," each "it's for your own good," was another bar in the cage of her childhood. She stared at the faded glow-in-the-dark stars on her ceiling, remnants of a younger, more optimistic self. They didn't twinkle with promise anymore; they mocked her with their unreachable distance. She'd never get out. Never. The walls of her room felt like they were closing in, squeezing the very breath from her.

##########################################################################################

# Example Task

## Current Chapter:
"Honestly, Mother, it's just a dance, not an invitation to join a biker gang," Ada argued, her voice tight with the familiar strain of a battle already lost. "Everyone is going. Everyone."
Mrs. Cranton pressed her lips into a thin line, her knitting needles clicking like a disapproving metronome. "Everyone isn't my daughter, Ada. The world isn't as safe as you think. A school dance, yes, but until when? And who will be there? These things matter."
"You mean you'll matter, hovering by the refreshment table, interrogating every boy who even looks in my direction?" Ada's sarcasm was a shield. "I'm sixteen, not six. You can't keep me bubble-wrapped forever!"
"As long as you are under my roof, I can and I will ensure your safety," Mrs. Cranton stated, her voice unyielding.
Ada slammed her bedroom door, the sound a pathetic punctuation mark to her crushing defeat. She threw herself onto the bed, the worn quilt offering no comfort, only a suffocating reminder of years spent in this room, dreaming of escape. It was hopeless. Utterly, achingly hopeless. Each parental decree, each "no," each "it's for your own good," was another bar in the cage of her childhood. She stared at the faded glow-in-the-dark stars on her ceiling, remnants of a younger, more optimistic self. They didn't twinkle with promise anymore; they mocked her with their unreachable distance. She'd never get out. Never. The walls of her room felt like they were closing in, squeezing the very breath from her.

## Next Paragraph Instructions:
Talk about how Ada ends up sneaking out the following night to go to her friend Carlie's place after bedtime. They are in their secret fort in the woods behind Carlie's house. They hear an eerie noise and go out to investigate. Make this part very suspenseful and scary.

## Next Paragraph Completion:
The following night, a crescent moon hung like a secret smile in the sky. Ada slipped from her window, a shadow against shadows, her heart a frantic drum against her ribs. Carlie was waiting, her grin flashing in the darkness. Their fort, a rickety haven of scavenged wood and old tarps deep in the woods behind Carlie's house, felt like the only free place on Earth. They whispered about the dance, about boys, about everything, until a sound sliced through the night. It wasn't an animal, not a branch falling. It was a low, mournful hum, an unearthly thrum that vibrated in their teeth. Goosebumps erupted on Ada's arms. "What was that?" Carlie breathed, eyes wide. They crept from the fort, the familiar woods suddenly alien and menacing, every rustle of leaves a potential threat.

##########################################################################################

# Example Task

## Current Chapter:
"Honestly, Mother, it's just a dance, not an invitation to join a biker gang," Ada argued, her voice tight with the familiar strain of a battle already lost. "Everyone is going. Everyone."
Mrs. Cranton pressed her lips into a thin line, her knitting needles clicking like a disapproving metronome. "Everyone isn't my daughter, Ada. The world isn't as safe as you think. A school dance, yes, but until when? And who will be there? These things matter."
"You mean you'll matter, hovering by the refreshment table, interrogating every boy who even looks in my direction?" Ada's sarcasm was a shield. "I'm sixteen, not six. You can't keep me bubble-wrapped forever!"
"As long as you are under my roof, I can and I will ensure your safety," Mrs. Cranton stated, her voice unyielding.
Ada slammed her bedroom door, the sound a pathetic punctuation mark to her crushing defeat. She threw herself onto the bed, the worn quilt offering no comfort, only a suffocating reminder of years spent in this room, dreaming of escape. It was hopeless. Utterly, achingly hopeless. Each parental decree, each "no," each "it's for your own good," was another bar in the cage of her childhood. She stared at the faded glow-in-the-dark stars on her ceiling, remnants of a younger, more optimistic self. They didn't twinkle with promise anymore; they mocked her with their unreachable distance. She'd never get out. Never. The walls of her room felt like they were closing in, squeezing the very breath from her.
The following night, a crescent moon hung like a secret smile in the sky. Ada slipped from her window, a shadow against shadows, her heart a frantic drum against her ribs. Carlie was waiting, her grin flashing in the darkness. Their fort, a rickety haven of scavenged wood and old tarps deep in the woods behind Carlie's house, felt like the only free place on Earth. They whispered about the dance, about boys, about everything, until a sound sliced through the night. It wasn't an animal, not a branch falling. It was a low, mournful hum, an unearthly thrum that vibrated in their teeth. Goosebumps erupted on Ada's arms. "What was that?" Carlie breathed, eyes wide. They crept from the fort, the familiar woods suddenly alien and menacing, every rustle of leaves a potential threat.

## Next Paragraph Instructions:
Describe what they see when they go outside: a bright disc that speeds away and they hear some leaves crackling behind their fort. Continue the suspense and end this part with Ada somehow being alone in the woods.

## Next Paragraph Completion:
Peering through the tangled branches, they saw it. Not a sound, but a light - a brilliant, impossible disc hovering silently above the tallest pines, bathing the clearing in an ethereal, pulsing glow. Then, with a speed that defied physics, it zipped upwards and vanished, leaving a void in the starlit sky. "Did you see that?" Ada whispered, her voice trembling. Before Carlie could answer, a sharp crackle of dry leaves erupted directly behind their fort. Instinctively, they spun. Nothing. Just the dense, inky blackness of the undergrowth. "Carlie?" Ada reached out, but her hand grasped only empty air. A sudden, disorienting silence pressed in. "Carlie!" she called, louder this time, a knot of pure dread tightening in her stomach. She was utterly alone.

##########################################################################################

# Example Task

## Current Chapter:
"Honestly, Mother, it's just a dance, not an invitation to join a biker gang," Ada argued, her voice tight with the familiar strain of a battle already lost. "Everyone is going. Everyone."
Mrs. Cranton pressed her lips into a thin line, her knitting needles clicking like a disapproving metronome. "Everyone isn't my daughter, Ada. The world isn't as safe as you think. A school dance, yes, but until when? And who will be there? These things matter."
"You mean you'll matter, hovering by the refreshment table, interrogating every boy who even looks in my direction?" Ada's sarcasm was a shield. "I'm sixteen, not six. You can't keep me bubble-wrapped forever!"
"As long as you are under my roof, I can and I will ensure your safety," Mrs. Cranton stated, her voice unyielding.
Ada slammed her bedroom door, the sound a pathetic punctuation mark to her crushing defeat. She threw herself onto the bed, the worn quilt offering no comfort, only a suffocating reminder of years spent in this room, dreaming of escape. It was hopeless. Utterly, achingly hopeless. Each parental decree, each "no," each "it's for your own good," was another bar in the cage of her childhood. She stared at the faded glow-in-the-dark stars on her ceiling, remnants of a younger, more optimistic self. They didn't twinkle with promise anymore; they mocked her with their unreachable distance. She'd never get out. Never. The walls of her room felt like they were closing in, squeezing the very breath from her.
The following night, a crescent moon hung like a secret smile in the sky. Ada slipped from her window, a shadow against shadows, her heart a frantic drum against her ribs. Carlie was waiting, her grin flashing in the darkness. Their fort, a rickety haven of scavenged wood and old tarps deep in the woods behind Carlie's house, felt like the only free place on Earth. They whispered about the dance, about boys, about everything, until a sound sliced through the night. It wasn't an animal, not a branch falling. It was a low, mournful hum, an unearthly thrum that vibrated in their teeth. Goosebumps erupted on Ada's arms. "What was that?" Carlie breathed, eyes wide. They crept from the fort, the familiar woods suddenly alien and menacing, every rustle of leaves a potential threat.
Peering through the tangled branches, they saw it. Not a sound, but a light - a brilliant, impossible disc hovering silently above the tallest pines, bathing the clearing in an ethereal, pulsing glow. Then, with a speed that defied physics, it zipped upwards and vanished, leaving a void in the starlit sky. "Did you see that?" Ada whispered, her voice trembling. Before Carlie could answer, a sharp crackle of dry leaves erupted directly behind their fort. Instinctively, they spun. Nothing. Just the dense, inky blackness of the undergrowth. "Carlie?" Ada reached out, but her hand grasped only empty air. A sudden, disorienting silence pressed in. "Carlie!" she called, louder this time, a knot of pure dread tightening in her stomach. She was utterly alone.

## Next Paragraph Instructions:
Have Ada meet the extraterrestrial in this chapter. It is a scary experience at first, but the sense of awe overwhelms her so she isn't scared. The alien itself is not scary, it's humanoid. Perhaps this was a mistake because after a moment, the alien grabs her arm and the flying disc returns in a flash. The chapter ends on a cliffhanger.

## Next Paragraph Completion:
A shadowy figure emerged from the deeper blackness beneath the ancient oaks. Ada's breath hitched, a silent scream trapped in her throat. It stepped into a sliver of moonlight, and the fear building within her gave way to an almost painful sense of wonder. It was tall, slender, its form undeniably humanoid, yet it shimmered with an internal light, its skin like polished moonstone. The air thrummed with an unspoken power. This wasn't a monster; it was...magnificent. Then, it moved, much quicker than Ada would've thought possible. A cool, surprisingly strong hand clamped around her arm. As a gasp escaped Ada's lips, the night sky above them tore open with silent, blinding light - the disc, returned in an instant, hovering directly overhead, its brilliance swallowing the stars.

##########################################################################################
'''

current_task = '''
# Current Task

## Current Chapter:
{current_chapter}

## Next Paragraph Instructions:
{instruction}

## Next Paragraph Completion:
'''

prompt_generic = task_desc_generic + examples + current_task
prompt_templated = task_desc_templated + examples + current_task
