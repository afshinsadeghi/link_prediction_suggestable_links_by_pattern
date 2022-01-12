# link_prediction_suggestable_links_by_pattern
a code to generate filtered triples for evaluating if these possible new links exist.

the pattern to generate these triples is:

 ## 1. first filter out those h that has more than threshod relations. (for example 2) (their h lists are longer than threshold)
     (in train_dic_t_r those lists which are longer than threshhold (for example 2), take them!)
    
## 2.from them, those h that have same r_t (one common relation and tail) extend their other r_t to each other
     to get those h that have use the value of train_dic_t_r. get those that have more than one h and take the values of the dict

#### step 2 can be extended to those that have more than one common relation.
