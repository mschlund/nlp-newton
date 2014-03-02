nlp-newton
==========

Supplementary material to our EACL 2014 paper "Fast and Accurate Unlexicalized Parsing via Structural Annotations".

Includes the python-script we used to annotate trees with dimension and to call the Stanford parser.
We made some (very minor) changes to the parser (the code is available on request of course).
For convenience we attached a compiled jar.

We modified some ~10 lines in the parser e.g. to enable CrossingBracket evaluation which was already there but not accessible. We are working on extracting a patch -- if you are interested in the modified code just write an email to schlund(at)model[dot]in[dot]tum[dot]de.

To run our code you need python nltk, the TüBa-D/Z treebank and the stanford parser
