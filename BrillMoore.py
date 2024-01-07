from collections import defaultdict
class BrillMoreAcc:
    
    
    '''
    This class assumes that every word has error probability depending on it's length
    
    '''
       
    def __init__(self, N=2, max_candidates=10):
         
        self.N = N
        self.max_candidates = max_candidates
        self.triples = []
        self.edit_dict = defaultdict(int)
        self.count_dict = defaultdict(int)
        self.alphabet = set()
        
    
    def fit(self, triples,error_rate = 0.97):
        """
        Inputs: 
            triples: list of tuples in (intended word,observed word,count of confusion) format
            error rate: Rate at which a character is mistakely typed        
        """        
        
        self.error_rate = error_rate
        self.triples = triples
        for x, w, count in triples:
            self.alphabet.update(set(x+w))
            alignment = self.align(x, w)
            edits = self._edits_from_alignment(alignment)
            #print(edits)            
            for edit in edits:
                self.edit_dict[edit] += count
        for edit in self.edit_dict.keys():
            self.count_dict[edit[0]] += self.edit_dict[edit]
        for edit in self.edit_dict.keys():
            self.edit_dict[edit] /= self.count_dict[edit[0]]
            
            
        
    def _edits_from_alignment(self, alignment):
        edits = []
#        print(len(alignment))
        for a, b in alignment:           
            if a != b:
                edits.append((a,b))
        return edits
    
    def probability_error(self, alignment):
        
    
    
    
    def likelihood(self,x,w):
        """
        Calculate the likelihood based on the model
        
        Input:
            x: Observed word
            w: Intended word
            
        Output:
            Probability that x is typed given intended word w
        
        """        

        
        #Calculate the probability of error in the word based on it's length
        #Longer words are more Error Prone
        #If x and w are same , return 
        if (x == w):
            return (1-(self.error_rate)**len(x))
        
        #Align the words
        alignment = self.align(x, w)                   
        edits = self._edits_from_alignment(alignment)
        
        #Multiply the probability for every edits
        prob = 1
        for edit in edits:
            prob*= self.edit_dict.get(edit,0.0005)  
            
        #Return probability that 'x' is typed given intended word 'w'
        return prob* (1-((self.error_rate)**len(x)))
    
    
    
    
    def align(self,x, w):
        """
        
        
        
        
        """        
        
        m = len(x)
        n = len(w)
        dp = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif x[i - 1] == w[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
        i = m
        j = n
        alignments = []
        while i > 0 and j > 0:
            if x[i - 1] == w[j - 1]:
                alignments.append((x[i - 1], w[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                alignments.append((x[i - 1], w[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                alignments.append((None, w[j - 1]))
                j -= 1
            else:
                alignments.append((x[i - 1], None))
                i -= 1
        while i > 0:
            alignments.append((x[i - 1], None))
            i -= 1
        while j > 0:
            alignments.append((None, w[j - 1]))
            j -= 1
        return alignments[::-1]   
    
    
    
class BrillMore:
    """
    This class assumes that every word has same probability of being an error word
    
    """
    def __init__(self, N=2, max_candidates=10):
        self.N = N
        self.max_candidates = max_candidates
        self.triples = []
        self.edit_dict = defaultdict(int)
        self.count_dict = defaultdict(int)
        self.alphabet = set()
        
    
    def fit(self, triples,error_rate = 0.8):
        self.error_rate = error_rate
        self.triples = triples
        for x, w, count in triples:
            self.alphabet.update(set(x+w))
            alignment = self.align(x, w)
            edits = self._edits_from_alignment(alignment)
            #print(edits)            
            for edit in edits:
                self.edit_dict[edit] += count
        for edit in self.edit_dict.keys():
            self.count_dict[edit[0]] += self.edit_dict[edit]
        for edit in self.edit_dict.keys():
            self.edit_dict[edit] /= self.count_dict[edit[0]]
            self.edit_dict[edit]*= (1-self.error_rate)
        
    def _edits_from_alignment(self, alignment):
        edits = []
        for a, b in alignment:           
            if a != b:
                edits.append((a,b))
        return edits
#         expanded_edits = []
#         for i, (a, b) in enumerate(edits):
#             for j in range(1, self.N+1):
#                 if i+j < len(edits):
#                     expanded_edits.append((a+edits[i+j][0], b+edits[i+j][1]))
#         return expanded_edits
    
    def likelihood(self,x,w):
        """
        
        
        """        
        prob = 1
        if (x == w):
            return self.error_rate
        alignment = self.align(x, w)        
        edits = self._edits_from_alignment(alignment)
        for edit in edits:
            prob*= self.edit_dict.get(edit,0.0005*(1-self.error_rate))            
        return prob
    
    
    
    
    def likelihood_from_list(self,x,W):
        likelihood_ordered_list = []
        for w in W:
            prob = 1
            if (x == w):
                prob = self.error_rate
            else:
                alignment = self.align(x, w)
                edits = self._edits_from_alignment(alignment)
                for edit in edits:
                    prob*= self.edit_dict.get(edit,0.0005*(1-self.error_rate))
            likelihood_ordered_list.append((prob,w))
        return sorted(likelihood_ordered_list)
            
        
        
    
    def transform(self, x):
        candidates = []
        candidate_probs = []
        for w in self.alphabet:
            alignment = self.align(x, w)
            edits = self._edits_from_alignment(alignment)
            prob = 1
            for a, b in edits:
                prob *= self.edit_dict.get((a,b), 0)
            if len(candidates) < self.max_candidates:
                candidates.append(w)
                candidate_probs.append(prob)
            elif prob > min(candidate_probs):
                min_idx = candidate_probs.index(min(candidate_probs))
                candidates[min_idx] = w
                candidate_probs[min_idx] = prob
        return candidates[candidate_probs.index(max(candidate_probs))]



    def align(self,x, w):
        m = len(x)
        n = len(w)
        dp = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif x[i - 1] == w[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
        i = m
        j = n
        alignments = []
        while i > 0 and j > 0:
            if x[i - 1] == w[j - 1]:
                alignments.append((x[i - 1], w[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                alignments.append((x[i - 1], w[j - 1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                alignments.append((None, w[j - 1]))
                j -= 1
            else:
                alignments.append((x[i - 1], None))
                i -= 1
        while i > 0:
            alignments.append((x[i - 1], None))
            i -= 1
        while j > 0:
            alignments.append((None, w[j - 1]))
            j -= 1
        return alignments[::-1]

    
    
    
    
