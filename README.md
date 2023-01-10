
# Observations
* Looks like the galactica-125m model uses a different tokenizer than GPT-J-6B. I tried testing the tokenizer separately from the rest of the system, and when I entered in "<|endoftext|>" it didn't give a single token as it should have, but it seemed to treat it as ordinary text. I need to fix this before I do training for real, but as a proof of concept, it wasn't a show-stopper so I continued without fixing this.
* The dataset I fine-tuned the model on consisted of a keyword, followed by a colon, followed by a quote. For example:
  * `love: When love is not madness, it is not love.`
* When I prompted the trained model with "hope:", the trained model completed it with gibberish that was in the same style as the fine-tuning data:
  * `hope: 10 years ago and the best day we've ever witnessed was to live up in a life, say me...`
* In contast, the original model reponded to the same quote with gibberish that sounded more "scientific":
  * `hope: The Rise of the New Millennium, Hallett). Since \(x_{a}^{2}<0\), it follows that \(|s(f)-1/x_{a}^{2}||>Lx_{a}^{3}/64y_{*}\); this implies that the...`
* I also tried prompting the trained model with something outside the fine-tuning set, and the result sounded more like the original model. When I prompted it with "virus:", I got back
  * `virus: a case study, Litt;  The effect of physical education on children's psychological well-being`
* This seems to show that the trained model still retained the ability to produce the style of its original training. However, there was some spill-over. When I prompted it with "phosphorylation:" I got back:
  `phosphorylation: How bad is it? How good is it? How important is the choice as to whether some of us are happy, or not, rather than those who are really satisfied.`
