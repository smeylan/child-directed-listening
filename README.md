# child-directed-listening

Analyses for "Child Directed Listening" project. 


# Setup

These assume you are working on macOS, and have Anaconda installed. 

```conda create -n child-grammar```

Download and install the appropriate R package here: https://repo.miserver.it.umich.edu/cran/

```pip3 install -r requirements.txt```

Run ```R``` to get an R session.

Then, following the recommendation/code of [1], run the following:

```

install.packages("rlang")

install.packages("lazyeval")

install.packages("ggplot2")

```

Now, back in Terminal, run

```pip3 install levenshtein scipy scikit-learn```

Finally, following an adaptation of the original installation directions here [2]:

```pip3 install --user ipykernel; python3 -m ipykernel install --user --name=child-grammar```

[1] (11/19/22) https://community.rstudio.com/t/install-rlang-package-issue/84072/2

[2] (11/19/22) https://github.com/smeylan/child-directed-listening