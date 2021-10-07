#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:01:41 2021

@author: boruttrpin
"""
import numpy as np
import streamlit as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(
page_title="f-div minimizer",
page_icon="🔦",
layout="wide"
)


def random_distrib(n_prop=3,mest=4):
    n=2**n_prop
    a=[]
    for i in range(n):
        a.append(np.random.uniform(0,1))
    a=[i/sum(a) for i in a]

    a=[0.0 if "e" in str(i) else float(str(i)[:mest]) for i in a]
    dif=1-sum(a)
    a[int(np.random.choice(n,1)[0])]+=dif
    return a

def chain_distrib(a=.7,p1=.8,q1=.4,p2=.6,q2=.3):
    # assuming a->b->c with pr(a) = a, pr(b|a)=p1, pr(b|~a)=q1, pr(c|b)=p2, pr(c|~b)=q2
    return [
        a*p1*p2,
        a*p1*(1-p2),
        a*(1-p1)*q2,
        a*(1-p1)*(1-q2),
        (1-a)*q1*p2,
        (1-a)*q1*(1-p2),
        (1-a)*(1-q1)*q2,
        (1-a)*(1-q1)*(1-q2)
        ]

def commoncause_distrib(a=.7,p1=.8,q1=.4,p2=.6,q2=.3):
    # assuming b<-a->c with pr(a) = a, pr(b|a)=p1, pr(b|~a)=q1, pr(c|a)=p2, pr(c|~a)=q2
    return [
        a*p1*p2,
        a*p1*(1-p2),
        a*(1-p1)*p2,
        a*(1-p1)*(1-p2),
        (1-a)*q1*q2,
        (1-a)*q1*(1-q2),
        (1-a)*(1-q1)*q2,
        (1-a)*(1-q1)*(1-q2)
        ]

def collider_distrib(a=.7,b=.6,p1=.8,p2=.4,p3=.6,p4=.3):
    # assuming a->c<-b with pr(a) = a, pr(b)=b, pr(c|a,b)=p1, pr(c|a,~b)=p2, pr(c|~a,b)=p3, pr(c|~a,~b)=p4
    return [
        a*b*p1,
        a*b*(1-p1),
        a*(1-b)*p2,
        a*(1-b)*(1-p2),
        (1-a)*b*p3,
        (1-a)*b*(1-p3),
        (1-a)*(1-b)*p4,
        (1-a)*(1-b)*(1-p4)
        ]

def f_of_x(x_state,divergence="kl"):
    if divergence == "kl":
        return x_state * np.log(x_state)
    if divergence == "hel":
        return (1-((x_state)**0.5))**2
    if divergence=="ikl":
        return -np.log(x_state)
    if divergence=="chisq":
        return (x_state - 1)**2

def update_by_minimization(prior="rand",posterior_a_b_c_if_a_c_if_b_c_if_a_c=[None, None,1,1,None,None],printing=1,plotting=1,rounding="Yes",decimround=3):
    # in case of skiing trip: assume a->b->c with a: exam, b: skiing, c: buying outfit
    # posterior_a_b_c_if_a_c_if_b_c_if_a_c is a list of fixed posterior a, b, c, if a then b, if b then c, if a then c. If None, then the parameter is not fixed in posterior.
    if prior=="rand":
        prior = random_distrib(3,4)
        while 0 in prior:
            prior = random_distrib(3,4)

    divergence="kl"
    divergence2="hel"
    divergence3="ikl"
    divergence4="chisq"
    obj_fun1= lambda x: prior[0]*f_of_x((x[0]/prior[0]),divergence)+prior[1]*f_of_x((x[1]/prior[1]),divergence)+prior[2]*f_of_x((x[2]/prior[2]),divergence)+prior[3]*f_of_x((x[3]/prior[3]),divergence)+prior[4]*f_of_x((x[4]/prior[4]),divergence)+prior[5]*f_of_x((x[5]/prior[5]),divergence)+prior[6]*f_of_x((x[6]/prior[6]),divergence)+prior[7]*f_of_x((x[7]/prior[7]),divergence)
    obj_fun2= lambda x: prior[0]*f_of_x((x[0]/prior[0]),divergence2)+prior[1]*f_of_x((x[1]/prior[1]),divergence2)+prior[2]*f_of_x((x[2]/prior[2]),divergence2)+prior[3]*f_of_x((x[3]/prior[3]),divergence2)+prior[4]*f_of_x((x[4]/prior[4]),divergence2)+prior[5]*f_of_x((x[5]/prior[5]),divergence2)+prior[6]*f_of_x((x[6]/prior[6]),divergence2)+prior[7]*f_of_x((x[7]/prior[7]),divergence2)
    obj_fun3= lambda x: prior[0]*f_of_x((x[0]/prior[0]),divergence3)+prior[1]*f_of_x((x[1]/prior[1]),divergence3)+prior[2]*f_of_x((x[2]/prior[2]),divergence3)+prior[3]*f_of_x((x[3]/prior[3]),divergence3)+prior[4]*f_of_x((x[4]/prior[4]),divergence3)+prior[5]*f_of_x((x[5]/prior[5]),divergence3)+prior[6]*f_of_x((x[6]/prior[6]),divergence3)+prior[7]*f_of_x((x[7]/prior[7]),divergence3)
    obj_fun4= lambda x: prior[0]*f_of_x((x[0]/prior[0]),divergence4)+prior[1]*f_of_x((x[1]/prior[1]),divergence4)+prior[2]*f_of_x((x[2]/prior[2]),divergence4)+prior[3]*f_of_x((x[3]/prior[3]),divergence4)+prior[4]*f_of_x((x[4]/prior[4]),divergence4)+prior[5]*f_of_x((x[5]/prior[5]),divergence4)+prior[6]*f_of_x((x[6]/prior[6]),divergence4)+prior[7]*f_of_x((x[7]/prior[7]),divergence4)


    bnds = [(0+1e-320, 1) for i in range(8)] # open lower bound

    a=posterior_a_b_c_if_a_c_if_b_c_if_a_c[0]
    b=posterior_a_b_c_if_a_c_if_b_c_if_a_c[1]
    c=posterior_a_b_c_if_a_c_if_b_c_if_a_c[2]
    if_a_then_b=posterior_a_b_c_if_a_c_if_b_c_if_a_c[3]
    if_b_then_c=posterior_a_b_c_if_a_c_if_b_c_if_a_c[4]
    if_a_then_c=posterior_a_b_c_if_a_c_if_b_c_if_a_c[5]
    cons = [{'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] - 1}]
    # every posterior distribution is constrained to sum up to 1
    if a!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[4] + x[5] + x[6] + x[7] -1+a}) # fixing a
    if b!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[2] + x[3] + x[6] + x[7] -1+b}) # fixing b
    if c!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[1] + x[3] + x[5] + x[7] -1+c}) # fixing c
    if if_a_then_b!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[0] + x[1] - if_a_then_b*(x[0] + x[1] + x[2] + x[3])})
    if if_b_then_c!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[0] + x[4] - if_b_then_c*(x[0] + x[1] + x[4] + x[5])})
    if if_a_then_c!=None:
        cons.append({'type': 'eq', 'fun': lambda x:  x[0] + x[2] - if_a_then_c*(x[0] + x[2] + x[4] + x[6])})




    res1  = minimize(obj_fun1, x0 =[0.5]*8,bounds=bnds,constraints=cons)
    res2  = minimize(obj_fun2, x0 =[0.5]*8,bounds=bnds,constraints=cons)
    res3  = minimize(obj_fun3, x0 =[0.5]*8,bounds=bnds,constraints=cons)
    res4  = minimize(obj_fun4, x0 =[0.5]*8,bounds=bnds,constraints=cons)
    if printing==1 and plotting==1:
        columnize=1
    else:
        columnize=0
    if printing==1:
        labs=["a,b,c","a,b,~c","a,~b,c","a,~b,~c","~a,b,c","~a,b,~c","~a,~b,c","~a,~b,~c"]
        tablelabs=["","prior","kl","hel","ikl","chisq"]
        adats=[0,prior[0]+prior[1]+prior[2]+prior[3],res1.x[0]+res1.x[1]+res1.x[2]+res1.x[3],res2.x[0]+res2.x[1]+res2.x[2]+res2.x[3],res3.x[0]+res3.x[1]+res3.x[2]+res3.x[3],res4.x[0]+res4.x[1]+res4.x[2]+res4.x[3]]
        bdats=[0,prior[0]+prior[1]+prior[4]+prior[5],res1.x[0]+res1.x[1]+res1.x[4]+res1.x[5],res2.x[0]+res2.x[1]+res2.x[4]+res2.x[5],res3.x[0]+res3.x[1]+res3.x[4]+res3.x[5],res4.x[0]+res4.x[1]+res4.x[4]+res4.x[5]]
        cdats=[0,prior[0]+prior[2]+prior[4]+prior[6],res1.x[0]+res1.x[2]+res1.x[4]+res1.x[6],res2.x[0]+res2.x[2]+res2.x[4]+res2.x[6],res3.x[0]+res3.x[2]+res3.x[4]+res3.x[6],res4.x[0]+res4.x[2]+res4.x[4]+res4.x[6]]


        if rounding=="yes":
            dats=np.vstack([labs,np.round(np.array(prior),decimround),np.round(res1.x,decimround),np.round(res2.x,decimround),np.round(res3.x,decimround),np.round(res4.x,decimround)]).T
            dats=np.vstack([tablelabs,dats,np.round(adats,decimround),np.round(bdats,decimround),np.round(cdats,decimround),["success?","",str(res1.success),str(res2.success),str(res3.success),str(res4.success)]])

        else:
            dats=np.vstack([labs,np.array(prior),res1.x,res2.x,res3.x,res4.x]).T
            dats=np.vstack([tablelabs,dats,adats,bdats,cdats,["success?","",str(res1.success),str(res2.success),str(res3.success),str(res4.success)]])

        dats[-4][0]="a"
        dats[-3][0]="b"
        dats[-2][0]="c"
        if columnize==0:
            st.table(dats)

    ys = [res1.x,res2.x,res3.x, res4.x]
    successes=[res1.success,res2.success,res3.success, res4.success]
    if plotting==1:
        fig=plt.figure()
        x = np.arange(1, 9)
        labels=["a,b,c","a,b,~c","a,~b,c","a,~b,~c","~a,b,c","~a,b,~c","~a,~b,c","~a,~b,~c"]
        plt.xticks(x, labels, rotation='vertical')
        plt.plot(x,prior,label="prior")
        plt.plot(x, ys[0],label="kl")
        plt.plot(x, ys[1],label="hel")
        plt.plot(x, ys[2],label="ikl")
        plt.plot(x,ys[3],label="chisq")
        plt.legend()
        if columnize==0:
            st.pyplot(fig)
            st.write("Note: lines often overlap completely.")
            st.write("minimized ... kl: Kullback Leibler divergence, hel: Hellinger distance, ikl: inverse Kullback Leibler, chisq: chi square divergence")
    if columnize==1:
        col1c,col2c=st.columns(2)
        with col1c:
            st.table(dats)
            datscsv = pd.DataFrame(dats)
            datscsv = datscsv.to_csv().encode('utf-8')
            st.download_button(
                label="Download values as CSV",
                data=datscsv,
                file_name='minfdiv.csv',
                mime='text/csv',)
        with col2c:
            st.pyplot(fig)
        st.write("Note: lines often overlap completely.")
        st.write("minimized ... kl: Kullback Leibler divergence, hel: Hellinger distance, ikl: inverse Kullback Leibler, chisq: chi square divergence")

    return prior,ys,successes

st.write(""" This is a tool to update 3 variable-networks by minimizing some central f-divergences; written by Borut Trpin""")
networktype = st.radio("Type of network?", ["random","chain","common cause","collider"])
if networktype == "chain":
    st.write("""assuming a->b->c
             \n
             pr(a)=a,
             pr(b|a)=p1,
             pr(b|~a)=q1,
             pr(c|b)=p2,
             pr(c|~b)=q2""")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        number_a1 = st.radio("Set prior probability of a?",["random", "yes"],key="1")
        if number_a1=="yes":
            number_a = st.slider("Choose value:",0.0,1.0,key="2")
        else:
            number_a = np.random.random()
        st.write("pr(a)="+str(number_a)[:6])
    with col2:
        number_p11 = st.radio("Set prior probability of p1?",["random", "yes"],key="3")
        if number_p11=="yes":
            number_p1 = st.slider("Choose value:",0.0,1.0,key="4")
        else:
            number_p1=np.random.random()
        st.write("pr(p1)="+str(number_p1)[:6])
    with col3:
        number_q11 = st.radio("Set prior probability of q1?",["random", "yes"],key="5")
        if number_q11=="yes":
            number_q1 = st.slider("Choose value:",0.0,1.0,key="6")
        else:
            number_q1=np.random.random()
        st.write("pr(q1)="+str(number_q1)[:6])
    with col4:
        number_p21 = st.radio("Set prior probability of p2?",["random", "yes"],key="7")
        if number_p21=="yes":
            number_p2 = st.slider("Choose value:",0.0,1.0,key="8")
        else:
            number_p2=np.random.random()
        st.write("pr(p2)="+str(number_p2)[:6])
    with col5:
        number_q21 = st.radio("Set prior probability of q2?",["random", "yes"],key="9")
        if number_q21=="yes":
            number_q2 = st.slider("Choose value:",0.0,1.0,key="10")
        else:
            number_q2=np.random.random()
        st.write("pr(q2)="+str(number_q2)[:6])
if networktype == "common cause":
    st.write("""assuming b<-a->c
             \n
             pr(a) = a,
             pr(b|a)=p1,
             pr(b|~a)=q1,
             pr(c|a)=p2,
             pr(c|~a)=q2""")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        number_a1 = st.radio("Set prior probability of a?",["random", "yes"],key="11")
        if number_a1=="yes":
            number_a = st.slider("Choose value:",0.0,1.0,key="12")
        else:
            number_a = np.random.random()
        st.write("pr(a)="+str(number_a)[:6])
    with col2:
        number_p11 = st.radio("Set prior probability of p1?",["random", "yes"],key="13")
        if number_p11=="yes":
            number_p1 = st.slider("Choose value:",0.0,1.0,key="14")
        else:
            number_p1=np.random.random()
        st.write("pr(p1)="+str(number_p1)[:6])
    with col3:
        number_q11 = st.radio("Set prior probability of q1?",["random", "yes"],key="15")
        if number_q11=="yes":
            number_q1 = st.slider("Choose value:",0.0,1.0,key="16")
        else:
            number_q1=np.random.random()
        st.write("pr(q1)="+str(number_q1)[:6])
    with col4:
        number_p21 = st.radio("Set prior probability of p2?",["random", "yes"],key="17")
        if number_p21=="yes":
            number_p2 = st.slider("Choose value:",0.0,1.0,key="18")
        else:
            number_p2=np.random.random()
        st.write("pr(p2)="+str(number_p2)[:6])
    with col5:
        number_q21 = st.radio("Set prior probability of q2?",["random", "yes"],key="19")
        if number_q21=="yes":
            number_q2 = st.slider("Choose value:",0.0,1.0,key="20")
        else:
            number_q2=np.random.random()
        st.write("pr(q2)="+str(number_q2)[:6])



if networktype == "collider":
    st.write("""assuming a->c<-b
             \n
             pr(a) = a,
             pr(b)=b,
             pr(c|a,b)=p1,
             pr(c|a,~b)=p2,
             pr(c|~a,b)=p3,
             pr(c|~a,~b)=p4""")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1:
        number_a1 = st.radio("Set prior probability of a?",["random", "yes"],key="21")
        if number_a1=="yes":
            number_a = st.slider("Choose value:",0.0,1.0,key="22")
        else:
            number_a = np.random.random()
        st.write("pr(a)="+str(number_a)[:6])
    with col2:
        number_b1 = st.radio("Set prior probability of b?",["random", "yes"],key="23")
        if number_b1=="yes":
            number_b = st.slider("Choose value:",0.0,1.0,key="24")
        else:
            number_b=np.random.random()
        st.write("pr(b)="+str(number_b)[:6])
    with col3:
        number_p11 = st.radio("Set prior probability of p1?",["random", "yes"],key="25")
        if number_p11=="yes":
            number_p1 = st.slider("Choose value:",0.0,1.0,key="26")
        else:
            number_p1=np.random.random()
        st.write("pr(p1)="+str(number_p1)[:6])
    with col4:
        number_p21 = st.radio("Set prior probability of p2?",["random", "yes"],key="27")
        if number_p21=="yes":
            number_p2 = st.slider("Choose value:",0.0,1.0,key="28")
        else:
            number_p2=np.random.random()
        st.write("pr(p2)="+str(number_p2)[:6])
    with col5:
        number_p31 = st.radio("Set prior probability of p3?",["random", "yes"],key="29")
        if number_p31=="yes":
            number_p3 = st.slider("Choose value:",0.0,1.0,key="30")
        else:
            number_p3=np.random.random()
        st.write("pr(p3)="+str(number_p3)[:6])
    with col6:
        number_p41 = st.radio("Set prior probability of p4?",["random", "yes"],key="31")
        if number_p41=="yes":
            number_p4 = st.slider("Choose value:",0.0,1.0,key="32")
        else:
            number_p4=np.random.random()
        st.write("pr(p4)="+str(number_p4)[:6])
constraintz=[]
col1a,col2a,col3a,col4a,col5a,col6a = st.columns(6)
with col1a:
    constraint_a1 = st.radio("Set posterior probability of a (constraint)?", ["no", "yes"],key="33")
    if constraint_a1=="yes":
        constraint_a= st.slider("Choose value:",0.0,1.0,key="34")
        st.write("pr*(a)="+str(constraint_a)[:6])
    else:
        constraint_a=None
constraintz.append(constraint_a)
with col2a:
    constraint_b1 = st.radio("Set posterior probability of b (constraint)?", ["no", "yes"],key="35")
    if constraint_b1=="yes":
        constraint_b= st.slider("Choose value:",0.0,1.0,key="36")
        st.write("pr*(b)="+str(constraint_b)[:6])
    else:
        constraint_b=None
constraintz.append(constraint_b)

with col3a:
    constraint_c1 = st.radio("Set posterior probability of c (constraint)?", ["no", "yes"],key="37")
    if constraint_c1=="yes":
        constraint_c= st.slider("Choose value:",0.0,1.0,key="38")
        st.write("pr*(c)="+str(constraint_c)[:6])
    else:
        constraint_c=None
constraintz.append(constraint_c)

with col4a:
    constraint_a_b1 = st.radio("Set posterior probability of b given a (constraint)?", ["no", "yes"],key="39")
    if constraint_a_b1=="yes":
        constraint_a_b= st.slider("Choose value:",0.0,1.0,key="40")
        st.write("pr*(b|a)="+str(constraint_a_b)[:6])
    else:
        constraint_a_b=None
constraintz.append(constraint_a_b)

with col5a:
    constraint_b_c1 = st.radio("Set posterior probability of c given b (constraint)?", ["no", "yes"],key="41")
    if constraint_b_c1=="yes":
        constraint_b_c= st.slider("Choose value:",0.0,1.0,key="42")
        st.write("pr*(c|b)="+str(constraint_b_c)[:6])
    else:
        constraint_b_c=None
constraintz.append(constraint_b_c)

with col6a:
    constraint_a_c1 = st.radio("Set posterior probability of c given a (constraint)?", ["no", "yes"],key="43")
    if constraint_a_c1=="yes":
        constraint_a_c= st.slider("Choose value:",0.0,1.0,key="44")
        st.write("pr*(c|a)="+str(constraint_a_c)[:6])
    else:
        constraint_a_c=None
constraintz.append(constraint_a_c)

col1b,col2b,col3b = st.columns(3)
with col1b:
    printing1 = st.radio("Write out the results?", ["no", "yes"],key="45")
with col2b:
    plotting1 = st.radio("Plot the results?", ["no", "yes"],key="46")
with col3b:
    rounding = st.radio("Rounding of values?",["no", "yes"],key="round")
    if rounding=="yes":
        decimround = st.slider("Number of decimals:",2,8,key="roundnr")
    else:
        decimround=2

if printing1=="yes":
    printing=1
else:
    printing=0
if plotting1=="yes":
    plotting=1
else:
    plotting=0


if networktype=="random":
    prior="rand"
elif networktype=="common cause":
    prior=commoncause_distrib(number_a,number_p1,number_q1,number_p2,number_q2)
elif networktype=="chain":
    prior=chain_distrib(number_a,number_p1,number_q1,number_p2,number_q2)
elif networktype=="collider":
    prior=collider_distrib(number_a,number_b,number_p1,number_p2,number_p3,number_p4)


if st.button('Perform an update by minimizing f-divergence'):
    st.write('To repeat, click again')
    st.write("You can change any parameters above.")
    st.write("Random parameters will be randomized on every rerun.")
    update_by_minimization(prior,constraintz,printing,plotting,rounding,decimround)
