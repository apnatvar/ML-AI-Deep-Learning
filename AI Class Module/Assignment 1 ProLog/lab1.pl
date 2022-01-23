suspect(luc).
suspect(paul).
suspect(alain).
suspect(bernard).
suspect(louis).

alibi(luc, bernard, tuesday).
alibi(paul, bernard, tuesday).
alibi(louis, luc, tuesday).
alibi(alain, luc, thursday).

nottrustworthy(alain).

revenge(paul, jean).
revenge(luc, jean).

beneficiary(bernard, jean).
beneficiary(jean, louis).

owesmoney(louis, jean).
owesmoney(luc, jean).

sawcommittcrime(jean, alain).

gun(luc).
gun(louis).
gun(alain).

specialinterest(X) :-
	sawcommittcrime(jean,X); beneficiary(X, jean); owesmoney(X, jean).

motive(X) :-
	specialinterest(X); revenge(X, jean). 
	
truealibi(X) :- 
	alibi(X, Y, tuesday), \+ nottrustworthy(Y).

murderer(X) :-
	motive(X), gun(X), \+ truealibi(X). 