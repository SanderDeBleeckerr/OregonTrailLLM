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

Ended up using: https://pixabay.com/music/traditional-country-wise-old-travelling-cowboy-guitar-drums-instrumental-308551/

Notice the game is a bit dull because you go right away to the next day not really showing the consequence of your action. Also the consequence of your action is already visible on the choise which is great for debugging but bad for immersion.

# Stage 6

Game runs too slow the second LLM call can be a faster model.

# Stage 7

Just copying oregon trail is not very creative.
Making it more playfully

# Stage 8

Cleaned up project: moved prompts to txt files so they re easier to read and manipulate

# Stage 9

Fixed turn taking slightly so effects are only shown / calculated after a decision is taken and the user gets to see the repercussions after his decision via an afterscreen.

# Stage 10 

The game is too easy, you can win by forcing the maximum of effort every turn.
Programmed a risk of getting a serious diseases if you push too hard.

# Stage 11

Refactored prompts

# Stage 12 

Split up prompt of adventures so sometimes you have the choice between travel fast, medium, stay. And sometimes you have the option to interact with another group. Both events are seperate prompts.

# Stage 13 

Made a post decision prompt to make up some story on how the results of the current day came to be.

# Stage 14

Found a bug I made in a prompt where the json became invalid , corrected it.

# Stage 15

Improved immersion by making every prompts textoutput more westernlike.

# Stage 16

The two bullet cost of scavenging sometimes leaks in the scavening result even when no bullets were found the scavenging result prompt starts talking about finding 2 bullets (cost of scavenging).

Fixed by removing abundant game state context from the prompt.