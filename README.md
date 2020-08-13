# Word-Embedding-Transformation
Word embedding transformation from scratch

# Final Accuracy
* Final Accuracy top1 (Mean of 10 evaluations):  26.382978723404256
* Final Accuracy top5 (Mean of 10 evaluations):  38.72340425531916 

# Translations
1. regresar
    * need
    * need, seek, help, give, get
1. cabra
    * goat
    * goat, pig, cow, donkey, turtle
1. parecer
    * can
    * can, might, want, seem, have
1. otras
    * other
    * other, these, both, others, many
1. encantado
    * wonderful
    * wonderful, strange, liked, fantastic, amazing
1. lengua
    * tongue
    * tongue, meat, pig, mouth, chicken
1. mike
    * mike
    * mike, jim, chris, dave, steve
1. hables
    * have
    * have, could, can, would, were
1. poder
    * power
    * power, use, the, need, ability
    
# Interesting things
* proper nouns (names) produce other proper nouns (in example mike)
* No appropriate translation of lengua even when language was in vectors
* hables translation:you speak (google translate) english does not have a single word to express this

# Credits
* [Problem Spec](https://docs.google.com/document/d/1GJfn2B6EI8JueDiBwzTAdD34d6pC99BSt6vldOmUCPQ/edit#)
* Based on [Paper](https://arxiv.org/abs/1309.4168)
