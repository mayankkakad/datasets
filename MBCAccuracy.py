import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
import warnings
from apyori import apriori
warnings.filterwarnings(action='ignore')

def preprocessing(dataframe):
    memes=[]
    games=[]
    movies=[]
    for i in range(len(dataframe)):
        temp=[]
        temp=dataframe.values[i,7].split(";")
        for j in range(len(temp)):
            temp[j]=temp[j].strip()
        memes.append(temp)
    for i in range(len(dataframe)):
        temp=[]
        temp=dataframe.values[i,8].split(";")
        for j in range(len(temp)):
            temp[j]=temp[j].strip()
        games.append(temp)
    for i in range(len(dataframe)):
        temp=[]
        temp=dataframe.values[i,9].split(";")
        for j in range(len(temp)):
            temp[j]=temp[j].strip()
        movies.append(temp)
    return dataframe,memes,games,movies

def new_dataset(dataframe,memes,games,movies):
    memecol=[]
    gamecol=[]
    moviecol=[]
    for i in range(len(memes)):
        memecol.append(memes[i][0])
        gamecol.append(games[i][0])
        moviecol.append(movies[i][0])
    memearr=np.array(memecol)
    gamearr=np.array(gamecol)
    moviearr=np.array(moviecol)
    newdf=dataframe
    newdf['Memes']=memearr
    newdf['Games']=gamearr
    newdf['Movies/Series']=moviearr
    return newdf

def decision_tree():
    memeTree=DecisionTreeClassifier()
    gameTree=DecisionTreeClassifier()
    movieTree=DecisionTreeClassifier()
    memeTree.fit(X_memetrain,Y_memetrain)
    gameTree.fit(X_gametrain,Y_gametrain)
    movieTree.fit(X_movietrain,Y_movietrain)
    memePred=memeTree.predict(X_memetest)
    gamePred=gameTree.predict(X_gametest)
    moviePred=movieTree.predict(X_movietest)
    memeDecPredList=list(memePred)
    gameDecPredList=list(gamePred)
    movieDecPredList=list(moviePred)
    finalDecMemePred=[]
    finalDecGamePred=[]
    finalDecMoviePred=[]
    for m in memeDecPredList:
        temp=list(memedict[m])
        temp.append(m)
        finalDecMemePred.append(temp)
    for g in gameDecPredList:
        temp=list(gamedict[g])
        temp.append(g)
        finalDecGamePred.append(temp)
    for mo in movieDecPredList:
        temp=list(moviedict[mo])
        temp.append(mo)
        finalDecMoviePred.append(temp)
    dec_meme_points=0
    dectot_meme_points=0
    for i in range(len(actualMemes)):
        actual=actualMemes[i]
        predicted=finalDecMemePred[i]
        dectot_meme_points=dectot_meme_points+1
        for j in actual:
            if j in predicted:
                dec_meme_points=dec_meme_points+1
                break
    dec_game_points=0
    dectot_game_points=0
    for i in range(len(actualGames)):
        actual=actualGames[i]
        predicted=finalDecGamePred[i]
        dectot_game_points=dectot_game_points+1
        for j in actual:
            if j in predicted:
                dec_game_points=dec_game_points+1
                break
    dec_movie_points=0
    dectot_movie_points=0
    for i in range(len(actualMovies)):
        actual=actualMovies[i]
        predicted=finalDecMoviePred[i]
        dectot_movie_points=dectot_movie_points+1
        for j in actual:
            if j in predicted:
                dec_movie_points=dec_movie_points+1
                break
    dec_given_points=dec_meme_points+dec_game_points+dec_movie_points
    dec_total_points=dectot_meme_points+dectot_game_points+dectot_movie_points
    dec_accuracy=(dec_given_points/dec_total_points)*100
    return dec_accuracy

def naive_bayes():
    memeNB=GaussianNB()
    gameNB=GaussianNB()
    movieNB=GaussianNB()
    memeNB.fit(X_memetrain,Y_memetrain)
    gameNB.fit(X_gametrain,Y_gametrain)
    movieNB.fit(X_movietrain,Y_movietrain)
    memeNBPred=memeNB.predict(X_memetest)
    gameNBPred=gameNB.predict(X_gametest)
    movieNBPred=movieNB.predict(X_movietest)
    memeNBPredList=list(memeNBPred)
    gameNBPredList=list(gameNBPred)
    movieNBPredList=list(movieNBPred)
    finalNBMemePred=[]
    finalNBGamePred=[]
    finalNBMoviePred=[]
    for m in memeNBPredList:
        temp=list(memedict[m])
        temp.append(m)
        finalNBMemePred.append(temp)
    for g in gameNBPredList:
        temp=list(gamedict[g])
        temp.append(g)
        finalNBGamePred.append(temp)
    for mo in movieNBPredList:
        temp=list(moviedict[mo])
        temp.append(mo)
        finalNBMoviePred.append(temp)
    nb_meme_points=0
    nbtot_meme_points=0
    for i in range(len(actualMemes)):
        actual=actualMemes[i]
        predicted=finalNBMemePred[i]
        nbtot_meme_points=nbtot_meme_points+1
        for j in actual:
            if j in predicted:
                nb_meme_points=nb_meme_points+1
                break
    nb_game_points=0
    nbtot_game_points=0
    for i in range(len(actualGames)):
        actual=actualGames[i]
        predicted=finalNBGamePred[i]
        nbtot_game_points=nbtot_game_points+1
        for j in actual:
            if j in predicted:
                nb_game_points=nb_game_points+1
                break
    nb_movie_points=0
    nbtot_movie_points=0
    for i in range(len(actualMovies)):
        actual=actualMovies[i]
        predicted=finalNBMoviePred[i]
        nbtot_movie_points=nbtot_movie_points+1
        for j in actual:
            if j in predicted:
                nb_movie_points=nb_movie_points+1
                break
    nb_given_points=nb_meme_points+nb_game_points+nb_movie_points
    nb_total_points=nbtot_meme_points+nbtot_game_points+nbtot_movie_points
    nb_accuracy=(nb_given_points/nb_total_points)*100
    return nb_accuracy

def svm_linear():
    memeSVM=SVC(kernel='linear',random_state=0)
    gameSVM=SVC(kernel='linear',random_state=0)
    movieSVM=SVC(kernel='linear',random_state=0)
    memeSVM.fit(X_memetrain,Y_memetrain)
    gameSVM.fit(X_gametrain,Y_gametrain)
    movieSVM.fit(X_movietrain,Y_movietrain)
    memeSVMPred=memeSVM.predict(X_memetest)
    gameSVMPred=gameSVM.predict(X_gametest)
    movieSVMPred=movieSVM.predict(X_movietest)
    memeSVMPredList=list(memeSVMPred)
    gameSVMPredList=list(gameSVMPred)
    movieSVMPredList=list(movieSVMPred)
    finalSVMMemePred=[]
    finalSVMGamePred=[]
    finalSVMMoviePred=[]
    for m in memeSVMPredList:
        temp=list(memedict[m])
        temp.append(m)
        finalSVMMemePred.append(temp)
    for g in gameSVMPredList:
        temp=list(gamedict[g])
        temp.append(g)
        finalSVMGamePred.append(temp)
    for mo in movieSVMPredList:
        temp=list(moviedict[mo])
        temp.append(mo)
        finalSVMMoviePred.append(temp)
    svm_meme_points=0
    svmtot_meme_points=0
    for i in range(len(actualMemes)):
        actual=actualMemes[i]
        predicted=finalSVMMemePred[i]
        svmtot_meme_points=svmtot_meme_points+1
        for j in actual:
            if j in predicted:
                svm_meme_points=svm_meme_points+1
                break
    svm_game_points=0
    svmtot_game_points=0
    for i in range(len(actualGames)):
        actual=actualGames[i]
        predicted=finalSVMGamePred[i]
        svmtot_game_points=svmtot_game_points+1
        for j in actual:
            if j in predicted:
                svm_game_points=svm_game_points+1
                break
    svm_movie_points=0
    svmtot_movie_points=0
    for i in range(len(actualMovies)):
        actual=actualMovies[i]
        predicted=finalSVMMoviePred[i]
        svmtot_movie_points=svmtot_movie_points+1
        for j in actual:
            if j in predicted:
                svm_movie_points=svm_movie_points+1
                break
    svm_given_points=svm_meme_points+svm_game_points+svm_movie_points
    svm_total_points=svmtot_meme_points+svmtot_game_points+svmtot_movie_points
    svm_accuracy=(svm_given_points/svm_total_points)*100
    return svm_accuracy

def svm_polynomial():
    memeSVMP=SVC(kernel='poly',random_state=0)
    gameSVMP=SVC(kernel='poly',random_state=0)
    movieSVMP=SVC(kernel='poly',random_state=0)
    memeSVMP.fit(X_memetrain,Y_memetrain)
    gameSVMP.fit(X_gametrain,Y_gametrain)
    movieSVMP.fit(X_movietrain,Y_movietrain)
    memeSVMPPred=memeSVMP.predict(X_memetest)
    gameSVMPPred=gameSVMP.predict(X_gametest)
    movieSVMPPred=movieSVMP.predict(X_movietest)
    memeSVMPPredList=list(memeSVMPPred)
    gameSVMPPredList=list(gameSVMPPred)
    movieSVMPPredList=list(movieSVMPPred)
    finalSVMPMemePred=[]
    finalSVMPGamePred=[]
    finalSVMPMoviePred=[]
    for m in memeSVMPPredList:
        temp=list(memedict[m])
        temp.append(m)
        finalSVMPMemePred.append(temp)
    for g in gameSVMPPredList:
        temp=list(gamedict[g])
        temp.append(g)
        finalSVMPGamePred.append(temp)
    for mo in movieSVMPPredList:
        temp=list(moviedict[mo])
        temp.append(mo)
        finalSVMPMoviePred.append(temp)
    svmp_meme_points=0
    svmptot_meme_points=0
    for i in range(len(actualMemes)):
        actual=actualMemes[i]
        predicted=finalSVMPMemePred[i]
        svmptot_meme_points=svmptot_meme_points+1
        for j in actual:
            if j in predicted:
                svmp_meme_points=svmp_meme_points+1
                break
    svmp_game_points=0
    svmptot_game_points=0
    for i in range(len(actualGames)):
        actual=actualGames[i]
        predicted=finalSVMPGamePred[i]
        svmptot_game_points=svmptot_game_points+1
        for j in actual:
            if j in predicted:
                svmp_game_points=svmp_game_points+1
                break
    svmp_movie_points=0
    svmptot_movie_points=0
    for i in range(len(actualMovies)):
        actual=actualMovies[i]
        predicted=finalSVMPMoviePred[i]
        svmptot_movie_points=svmptot_movie_points+1
        for j in actual:
            if j in predicted:
                svmp_movie_points=svmp_movie_points+1
                break
    svmp_given_points=svmp_meme_points+svmp_game_points+svmp_movie_points
    svmp_total_points=svmptot_meme_points+svmptot_game_points+svmptot_movie_points
    svmp_accuracy=(svmp_given_points/svmp_total_points)*100
    return svmp_accuracy

def logistic_regression():
    memeReg=LogisticRegression(random_state=0)
    gameReg=LogisticRegression(random_state=0)
    movieReg=LogisticRegression(random_state=0)
    memeReg.fit(X_memetrain,Y_memetrain)
    gameReg.fit(X_gametrain,Y_gametrain)
    movieReg.fit(X_movietrain,Y_movietrain)
    memeRegPred=memeReg.predict(X_memetest)
    gameRegPred=gameReg.predict(X_gametest)
    movieRegPred=movieReg.predict(X_movietest)
    memeLRPredList=list(memeRegPred)
    gameLRPredList=list(gameRegPred)
    movieLRPredList=list(movieRegPred)
    finalLRMemePred=[]
    finalLRGamePred=[]
    finalLRMoviePred=[]
    for m in memeLRPredList:
        temp=list(memedict[m])
        temp.append(m)
        finalLRMemePred.append(temp)
    for g in gameLRPredList:
        temp=list(gamedict[g])
        temp.append(g)
        finalLRGamePred.append(temp)
    for mo in movieLRPredList:
        temp=list(moviedict[mo])
        temp.append(mo)
        finalLRMoviePred.append(temp)
    lr_meme_points=0
    lrtot_meme_points=0
    for i in range(len(actualMemes)):
        actual=actualMemes[i]
        predicted=finalLRMemePred[i]
        lrtot_meme_points=lrtot_meme_points+1
        for j in actual:
            if j in predicted:
                lr_meme_points=lr_meme_points+1
                break
    lr_game_points=0
    lrtot_game_points=0
    for i in range(len(actualGames)):
        actual=actualGames[i]
        predicted=finalLRGamePred[i]
        lrtot_game_points=lrtot_game_points+1
        for j in actual:
            if j in predicted:
                lr_game_points=lr_game_points+1
                break
    lr_movie_points=0
    lrtot_movie_points=0
    for i in range(len(actualMovies)):
        actual=actualMovies[i]
        predicted=finalLRMoviePred[i]
        lrtot_movie_points=lrtot_movie_points+1
        for j in actual:
            if j in predicted:
                lr_movie_points=lr_movie_points+1
                break
    lr_given_points=lr_meme_points+lr_game_points+lr_movie_points
    lr_total_points=lrtot_meme_points+lrtot_game_points+lrtot_movie_points
    lr_accuracy=(lr_given_points/lr_total_points)*100
    return lr_accuracy

def runApriori(memedict,gamedict,moviedict,memes,games,movies):
    minsup=0.12
    mincon=0.7
    minlift=2
    assoc_rules_memes=list(apriori(memes,min_support=minsup,min_confidence=mincon,min_lift=minlift))
    assoc_rules_games=list(apriori(games,min_support=minsup,min_confidence=mincon,min_lift=minlift))
    assoc_rules_movies=list(apriori(movies,min_support=minsup,min_confidence=mincon,min_lift=minlift))
    for i in assoc_rules_memes:
        sets=list(i[0])
        temp=list(memedict[str(sets[0])])
        temp.extend(sets[1:])
        temp=list(set(temp))
        memedict[sets[0]]=temp
    for i in assoc_rules_games:
        sets=list(i[0])
        temp=list(gamedict[str(sets[0])])
        temp.extend(sets[1:])
        temp=list(set(temp))
        gamedict[sets[0]]=temp
    for i in assoc_rules_movies:
        sets=list(i[0])
        temp=list(moviedict[str(sets[0])])
        temp.extend(sets[1:])
        temp=list(set(temp))
        moviedict[sets[0]]=temp
    return memedict,gamedict,moviedict

decwin=0
nbwin=0
svmwin=0
svmpwin=0
lrwin=0

for k in range(100):
    moods=['1. Anxiety','2. Anger','3. Hopelessness','4. Perpetual/Long-term Boredom / Tiredness','5. Unreasonable/Unexplained Sadness']
    tmoods=['Timestamp','1. Anxiety','2. Anger','3. Hopelessness','4. Perpetual/Long-term Boredom / Tiredness','5. Unreasonable/Unexplained Sadness']
    dataframe=pd.read_csv('https://raw.githubusercontent.com/mayankkakad/datasets/main/mbcdataset.csv')
    dataframe,memes,games,movies=preprocessing(dataframe)
    newdf=new_dataset(dataframe,memes,games,movies)
    X=newdf[tmoods].values
    Y_meme=newdf['Memes'].values
    Y_game=newdf['Games'].values
    Y_movie=newdf['Movies/Series'].values
    X_tmemetrain,X_tmemetest,Y_memetrain,Y_memetest=train_test_split(X,Y_meme,test_size=0.25)
    X_tgametrain,X_tgametest,Y_gametrain,Y_gametest=train_test_split(X,Y_game,test_size=0.25)
    X_tmovietrain,X_tmovietest,Y_movietrain,Y_movietest=train_test_split(X,Y_movie,test_size=0.25)
    X_memetrain=[]
    for tmx in X_tmemetrain:
        tmx=np.delete(tmx,[0])
        X_memetrain.append(tmx)
    X_memetrain=np.array(X_memetrain)
    X_memetrain=X_memetrain.astype('int64')
    X_gametrain=[]
    for tgx in X_tgametrain:
        tgx=np.delete(tgx,[0])
        X_gametrain.append(tgx)
    X_gametrain=np.array(X_gametrain)
    X_gametrain=X_gametrain.astype('int64')
    X_movietrain=[]
    for tmox in X_tmovietrain:
        tmox=np.delete(tmox,[0])
        X_movietrain.append(tmox)
    X_movietrain=np.array(X_movietrain)
    X_movietrain=X_movietrain.astype('int64')
    X_memetest=[]
    for tmx in X_tmemetest:
        tmx=np.delete(tmx,[0])
        X_memetest.append(tmx)
    X_memetest=np.array(X_memetest)
    X_memetest=X_memetest.astype('int64')
    X_gametest=[]
    for tgx in X_tgametest:
        tgx=np.delete(tgx,[0])
        X_gametest.append(tgx)
    X_gametest=np.array(X_gametest)
    X_gametest=X_gametest.astype('int64')
    X_movietest=[]
    for tmox in X_tmovietest:
        tmox=np.delete(tmox,[0])
        X_movietest.append(tmox)
    X_movietest=np.array(X_movietest)
    X_movietest=X_movietest.astype('int64')
    memedict={'Doggo':[],'Bollywood':[],'Food':[],'Sports':[],'Travel':[],'Political':[],'Dark':[]}
    gamedict={'Action':[],'Multiplayer':[],'Arcade':[],'Sports':[],'Racing':[],'Puzzle':[],'Adventure':[]}
    moviedict={'Drama':[],'Comedy':[],'Horror':[],'Action':[],'Romance':[],'Science fiction':[],'Animation':[],'Thriller':[],'Crime':[],'Biography':[]}
    memedict,gamedict,moviedict=runApriori(memedict,gamedict,moviedict,memes,games,movies)
    df=pd.read_csv('https://raw.githubusercontent.com/mayankkakad/datasets/main/mbcdataset.csv')
    actualMemes=[]
    actualGames=[]
    actualMovies=[]
    for tmx in X_tmemetest:
        tempvar=list(df.loc[(df['Timestamp']==tmx[0])&(df['1. Anxiety']==tmx[1])&(df['2. Anger']==tmx[2])&(df['3. Hopelessness']==tmx[3])&(df['4. Perpetual/Long-term Boredom / Tiredness']==tmx[4])&(df['5. Unreasonable/Unexplained Sadness']==tmx[5])]['Memes'])
        tempvar=tempvar[0].split(';')
        actualMemes.append(tempvar)
    for tgx in X_tgametest:
        tempvar=list(df.loc[(df['Timestamp']==tgx[0])&(df['1. Anxiety']==tgx[1])&(df['2. Anger']==tgx[2])&(df['3. Hopelessness']==tgx[3])&(df['4. Perpetual/Long-term Boredom / Tiredness']==tgx[4])&(df['5. Unreasonable/Unexplained Sadness']==tgx[5])]['Games'])
        tempvar=tempvar[0].split(';')
        actualGames.append(tempvar)
    for tmox in X_tmovietest:
        tempvar=list(df.loc[(df['Timestamp']==tmox[0])&(df['1. Anxiety']==tmox[1])&(df['2. Anger']==tmox[2])&(df['3. Hopelessness']==tmox[3])&(df['4. Perpetual/Long-term Boredom / Tiredness']==tmox[4])&(df['5. Unreasonable/Unexplained Sadness']==tmox[5])]['Movies/Series'])
        tempvar=tempvar[0].split(';')
        actualMovies.append(tempvar)
    temp=[decision_tree(),naive_bayes(),svm_linear(),svm_polynomial(),logistic_regression()]
    ind=temp.index(max(temp))
    if ind==0:
        decwin=decwin+1
    elif ind==1:
        nbwin=nbwin+1
    elif ind==2:
        svmwin=svmwin+1
    elif ind==3:
        svmpwin=svmpwin+1
    elif ind==4:
        lrwin=lrwin+1

temp=[decwin,nbwin,svmwin,lrwin]
print('\nScores: -\n')
print('Decision tree:',decwin)
print('Naive Bayes:',nbwin)
print('SVM Linear Kernel',svmwin)
print('SVM Polynomial Kernel: ',svmpwin)
print('Logistic Regression: ',lrwin)
winner=temp.index(max(temp))
if winner==0:
    print('\nWinner: Decision Tree')
elif winner==1:
    print('\nWinner: Naive Bayes')
elif winner==2:
    print('\nWinner: SVM Linear Kernel')
elif winner==3:
    print('\nWinner: SVM Polynomial Kernel')
elif winner==4:
    print('\nWinner: Logistic Regression')