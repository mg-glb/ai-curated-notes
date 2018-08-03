#Create the dictionary with all the options
av="The Avengers"
tm="The Martin"
gg="Guardians of the Galaxy"
eot="Edge of Tomorrow"
tmr="The Maze Runner"
u="Unbroken"
a={av: 3.0, tm: 4.0, gg: 3.5, eot: 5.0, tmr: 3.0}
b={av: 3.0, tm: 4.0, gg: 4.0, eot: 3.0, tmr: 3.0, u: 4.5}
c={tm: 1.0, gg: 2.0, eot: 5.0, tmr: 4.5, u: 2.0}
data = {"Jacob": a,"Jane": b,"Jim": c}

#Now we'll be using this similar_mean function to compute the average difference in rating.
#This will tell if ratings on the movies are similar, or not.
#We'll have threshold of 1 rating or less to consider the recommendation an adequate one.
def similar_mean(same_movies, user1, user2, dataset):
    total = 0
    for movie in same_movies:
        total += abs(dataset.get(user1).get(movie) - dataset.get(user2).get(movie))
    return total/len(same_movies)
pairs=[]
pairs.append(("Jacob","Jane"))
pairs.append(("Jacob","Jim"))
common_movies=[]
#We can also find the movies that have been watched by Jacob and Jane by using the intersection of both sets.
#We will save this as a list in common_movies for later use.
common_movies.append(list(set(data.get(pairs[0][0])).intersection(data.get(pairs[0][1]))))
#Now the same for Jacob and Jim
common_movies.append(list(set(data.get(pairs[1][0])).intersection(data.get(pairs[1][1]))))
recommendation=[]
#Possible recommendations are the movies that Jane has watched and the ones Jacob has not.
recommendation.append(list(set(data.get(pairs[0][1])).difference(data.get(pairs[0][0]))))
#Do the same thing for Jacob and Jim
recommendation.append(list(set(data.get(pairs[1][0])).difference(data.get(pairs[1][1]))))
score=[]
#Get the average difference in score between Jacob and Jane.
score.append(similar_mean(common_movies[0], pairs[0][0], pairs[0][1], data))
#Get the average difference in score between Jacob and Jim.
score.append(similar_mean(common_movies[1], pairs[1][0], pairs[1][1], data))

#Finally we use the difference in score to determine whether to recommend the films to each person.
for recs,sco,pair in zip(recommendation,score,pairs):
    for rec in recs:
        if sco < 1:
            print("%s should take %s's recommendation to view %s" % (pair[1],pair[0],rec))
        else:
            print("%s should not take %s's recommendation to view %s" % (pair[1],pair[0],rec))

            