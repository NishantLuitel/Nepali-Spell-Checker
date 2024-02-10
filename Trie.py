import textdistance

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()


    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
        
    def insert_list(self,lis):
        for w in lis:
            self.insert(w)
        return self

    def search(self, word, max_distance):
        current = self.root
        queue = [(current, "", 0)]
        found_words = []
        
        while queue:
            current, current_word, distance = queue.pop(0)
            if current.is_end_of_word:
                if distance <= max_distance:
                    found_words.append(current_word)
            for char, node in current.children.items():
                new_word = current_word + char
                new_distance = textdistance.levenshtein.distance(word, new_word)
                if new_distance <= int(abs(len(new_word) - len(word)))+2:
                    queue.append((node, new_word, new_distance))
        return found_words