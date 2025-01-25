import numpy as np

dataset = [
    # Spam Examples (Financial Scams)
    ("Spam", "Win $100,000 instantly!"),
    ("Spam", "Claim your free million-dollar prize now!"),
    ("Spam", "Guaranteed investment with 500% returns"),
    ("Spam", "Urgent: Bank transfer required"),
    ("Spam", "You've inherited $5 million from a distant relative"),
    ("Spam", "Exclusive crypto investment opportunity"),
    ("Spam", "Get rich quick - guaranteed method"),
    ("Spam", "Instant cash loan - no credit check"),
    ("Spam", "Offshore banking secrets revealed"),
    ("Spam", "Become a millionaire in 30 days"),
    
    # Spam Examples (Product Offers)
    ("Spam", "Exclusive discount - 90% off all products"),
    ("Spam", "Limited time offer: Buy one get ten free"),
    ("Spam", "Miracle weight loss pill - lose 30 pounds in 30 days"),
    ("Spam", "Revolutionary anti-aging cream"),
    ("Spam", "Cheap medications without prescription"),
    ("Spam", "Free iPhone - claim now!"),
    ("Spam", "Unbelievable gadget sale"),
    ("Spam", "Miracle supplement - boost your health"),
    ("Spam", "Exclusive beauty product clearance"),
    ("Spam", "Limited stock - must-have electronics"),
    
    # Spam Examples (Sexual Content)
    ("Spam", "Hot singles in your area"),
    ("Spam", "Adult content special offer"),
    ("Spam", "Enlarge your performance now"),
    ("Spam", "Dating secrets revealed"),
    ("Spam", "Intimate connections guaranteed"),
    
    # Spam Examples (Tech Scams)
    ("Spam", "Your computer is infected"),
    ("Spam", "Microsoft support urgent message"),
    ("Spam", "Claim your free software now"),
    ("Spam", "Urgent security update required"),
    ("Spam", "Hack-proof your system today"),
    
    # Ham Examples (Professional Communication)
    ("Ham", "Quarterly financial report attached"),
    ("Ham", "Meeting minutes for project review"),
    ("Ham", "Client presentation draft for feedback"),
    ("Ham", "Team performance metrics Q3"),
    ("Ham", "Budget allocation for next fiscal year"),
    ("Ham", "Project status update"),
    ("Ham", "Vendor contract review"),
    ("Ham", "Compliance documentation needed"),
    ("Ham", "Strategic planning meeting agenda"),
    ("Ham", "Quarterly earnings report"),
    
    # Ham Examples (Personal Communication)
    ("Ham", "Dinner plans this weekend?"),
    ("Ham", "Can we reschedule our call?"),
    ("Ham", "Happy birthday! Hope you're doing well"),
    ("Ham", "Sending you the family photos"),
    ("Ham", "What time works for your schedule?"),
    ("Ham", "Catching up over coffee"),
    ("Ham", "Weekend plans"),
    ("Ham", "Travel itinerary confirmation"),
    ("Ham", "Movie recommendation"),
    ("Ham", "Gift ideas for mom"),
    
    # Ham Examples (Academic/Educational)
    ("Ham", "Research paper draft for review"),
    ("Ham", "Thesis submission guidelines"),
    ("Ham", "Scholarship application details"),
    ("Ham", "Conference presentation schedule"),
    ("Ham", "Study group meeting time"),
    ("Ham", "Internship opportunity details"),
    ("Ham", "Academic journal submission"),
    ("Ham", "Research funding proposal"),
    ("Ham", "Lecture notes sharing"),
    ("Ham", "Graduate program application"),
    
    # Mixed/Borderline Examples
    ("Spam", "Exclusive webinar: Make money online"),
    ("Ham", "Upcoming professional development workshop"),
    ("Spam", "Get rich working from home"),
    ("Ham", "Remote work policy update"),
    ("Spam", "Work from home - earn $5000 weekly"),
    ("Ham", "Freelance job opportunity"),
    ("Spam", "Guaranteed online income"),
    ("Ham", "Professional networking event"),
    ("Spam", "Passive income secrets"),
    ("Ham", "Career development seminar")
]

# Labels: Spam = 1, Ham = 0
labels = np.array([1 if label == "Spam" else 0 for label, _ in dataset])

# Tokenize messages and build vocabulary
def tokenize(messages):
    vocab = {}
    tokenized_messages = []
    for message in messages:
        tokens = message.split()
        tokenized_message = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)  # Assign unique index
            tokenized_message.append(vocab[token])
        tokenized_messages.append(tokenized_message)
    return np.array(tokenized_messages, dtype=object), vocab

messages = [message for _, message in dataset]
tokenized_messages, vocab = tokenize(messages)

# print("Vocabulary:", vocab)
# print("Tokenized Messages:", tokenized_messages)


def bag_of_words(tokenized_messages, vocab_size):
    bow_matrix = np.zeros((len(tokenized_messages), vocab_size))
    for i, message in enumerate(tokenized_messages):
        for word_index in message:
            bow_matrix[i, word_index] += 1
    return bow_matrix

vocab_size = len(vocab)
bow_matrix = bag_of_words(tokenized_messages, vocab_size)

# print("Bag-of-Words Matrix:")
# print(bow_matrix)




# Separate spam and ham messages
spam_messages = bow_matrix[labels == 1]
ham_messages = bow_matrix[labels == 0]

# Prior probabilities
P_spam = np.sum(labels == 1) / len(labels)
P_ham = 1 - P_spam

# Word probabilities
P_word_given_spam = (np.sum(spam_messages, axis=0) + 1) / (np.sum(spam_messages) + vocab_size)
P_word_given_ham = (np.sum(ham_messages, axis=0) + 1) / (np.sum(ham_messages) + vocab_size)

# print("P(Word | Spam):", P_word_given_spam)
# print("P(Word | Ham):", P_word_given_ham)



def classify_message(new_message, vocab, P_word_given_spam, P_word_given_ham, P_spam, P_ham):
    # Tokenize new message
    tokenized_message = [vocab[word] for word in new_message.split() if word in vocab]
    bow_vector = np.zeros(len(vocab))
    for word_index in tokenized_message:
        bow_vector[word_index] += 1
    
    # Calculate probabilities
    spam_score = np.log(P_spam) + np.sum(bow_vector * np.log(P_word_given_spam))
    ham_score = np.log(P_ham) + np.sum(bow_vector * np.log(P_word_given_ham))
    
    return "Spam" if spam_score > ham_score else "Ham"

new_message = input('Enter a message: ')
result = classify_message(new_message, vocab, P_word_given_spam, P_word_given_ham, P_spam, P_ham)
print(f"The message '{new_message}' is classified as: {result}")
