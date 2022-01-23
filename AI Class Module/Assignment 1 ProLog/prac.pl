parent(pam,bob).
parent(tom,bob).
parent(tom,liz).
parent(bob,ann).
parent(bob,pat).
parent(pat,jim).

female(pam).
female(ann).
female(pat).
female(liz).

male(tom).
male(bob).
male(jim).

father(X,Y) :-
	male(X), parent(X,Y).
mother(X,Y) :- 
	female(X), parent(X,Y).
brother(X,Y) :-
	male(X), parent(Z,X), parent(Z,Y), X \== Y.
sister(X,Y) :-
	female(X), parent(Z,X), parent(Z,Y), X \== Y.
grandparent(X,Y) :-
	parent(X,Z), parent(Z,Y).
grandfather(X,Y) :-
	male(X), grandparent(X,Y).
grandmother(X,Y) :-
	female(X), grandparent(X,Y).
