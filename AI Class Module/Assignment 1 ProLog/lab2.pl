student(name(john, smith), [mat101, csi108, csi148, csi270]). 
student(name(jim, roy), []).
student(name(jane, brown), [mat101, csi108]). 
student(name(emily, white), [mat101, csi108, mat246]). 
student(name(emma, smith), [mat101, csi108, csi148, mat246]).

prereq(csi108, []).
prereq(csi148, [csi108, mat101]).
prereq(csi238, [csi148]).
prereq(csi260, [csi108]).
prereq(csi270, [csi148]).
prereq(csi310, [[csi260, csi270], [sta107, sta205, sta255]]). 
prereq(csi324, [[csi148], [csi238, mat246]]).

took(Name,Course):- 
	student(name(Name,_),Listofcourses), member(Course,Listofcourses).

checking_prereq(Init,[Init|Init]).
is_prereq(Course1,Course2):- 
	prereq(Course1, L), checking_prereq(Course2,L).
checking_prereq(Course1,[Course2|Init]):- 
	is_list(Course2),member(Course1,Course2).
checking_prereq(Course1,[Course2|Recur]):- 
	\+ Course1=Course2, checking_prereq(Course1,Recur).

check_for_course(Init,[Init]).
can_take(Name,Course):- 
	\+(took(Name,Course)), check_for_course(Name,Course).
check_for_course(Name,Course):-
	prereq(Course,L), check_for_course(Name,L).
check_for_course(Name,[H|T]):- 
	taken_one(Name,H), check_for_course(Name,T).
taken_one(Name,C):- 
	\+(is_list(C)), took(Name,C).
taken_one(Name,[H|T]):- 
	took(Name,H); taken_one(Name,T).

check_for_list_of_Courses(Init,[Init]).
check_for_list_of_Courses(Name,[Course|Other_Courses]):- 
	can_take(Name,Course), check_for_list_of_Courses(Name,Other_Courses).