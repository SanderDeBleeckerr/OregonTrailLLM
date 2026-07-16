# Stage 1

Default phi-3 model is not interesting, tried qwen3:30b model which failed due to reasoning intermingling with expected output.

# stage 2

Disabled reasoning but this did not suffice so I made a list of suitable models.


# stage 3

Context window is too short for gemma4:e4b for keeping +10 turns interesting enough.
Changed to different model and removed ollama defaults context limit.
ollama pull hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF:IQ4_XS

# Stage 4

The rewards dont seem fitting for the choices very often, saving an ox by resting costs you an ox according to the game. The LLM call will now be 1 LLM call for the options and 1 for the rewards allowing more prompt tuning for the rewards.

# Stage 5

Considering often showing a graphical image via a seperate model that displays the choice to take.

ollama pull x/z-image-turbo

Considering adding sounds for ambience noise or when a choise creates suspsense.

https://pixabay.com/music

Notice the game is a bit dull because you go right away to the next day not really showing the consequence of your action. Also the consequence of your action is already visible on the choise which is great for debugging but bad for immersion.

# Stage 6

Game runs too slow the second LLM call can be a faster model.

# Stage 7

Just copying oregon trail is not very creative.
Moving to something playfully funny like you're in past times trying to move your cart to the university of Harlem.

# Stage 8

Cleaned up project moved prompts to txt files so they re easier to read and manipulate