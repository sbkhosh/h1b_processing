#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import re
import urllib
import itertools
import asyncio
import grequests
import concurrent.futures
import bs4 as bs
import requests
import glob
import datetime
import lxml.html as lh
import dateutil
from numpy.random import rand
from matplotlib import style
from heapq import nlargest
from pandas import DataFrame
from datetime import datetime
from aiohttp import ClientSession
from random import random
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

pd.options.mode.chained_assignment = None 

params = {
    'scrape': True,
    'company': 'Facebook',
    'index': 'location',
    'distr': 'thread'
}

def get_headers(df):
    return(df.columns.values)

def read_data(path):
    df = pd.read_csv(path,sep=',')
    return(df)

def view_data(df):
    print(df.head(20))
  
def get_info(df):  
    df.info()
    df.describe()
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
   
def show_cmtx(df):
    corr_matrix = df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=0.5)
    plt.show()

def omit_by(dct, predicate=lambda x: x!=0):
    return({k:v for k,v in dct.items() if predicate(v)})
    
def check_missing_data(df):
    res = df.isnull().sum().sort_values(ascending=False)
    print(res)
    
    if(sum(res.values) != 0):
        kv_nz = omit_by(res)
        for el in kv_nz.keys():        
            print(df[df[str(el)].isnull()])

def write_to(df,name,flag):
    try:
        if(flag=="csv"):
            df.to_csv(str(name)+".csv",index=False)
        elif(flag=="html"):
            df.to_html(str(name)+"html",index=False)
    except:
        print("No other types supported")
        
def process_cols(df):
    df.dropna(inplace=True)
    
    df['submit date'] = df['submit date'].apply(dateutil.parser.parse)
    df['yr_sub'] = df['submit date'].apply(lambda x: x.year)
    df['mo_sub'] = df['submit date'].apply(lambda x: x.month)
    df['dy_sub'] = df['submit date'].apply(lambda x: x.day)

    df['start date'] = df['start date'].apply(dateutil.parser.parse)
    df['yr_str'] = df['start date'].apply(lambda x: x.year)
    df['mo_str'] = df['start date'].apply(lambda x: x.month)
    df['dy_str'] = df['start date'].apply(lambda x: x.day)

    df['delta date'] = df['start date'] - df['submit date']

    df['mo_sub'] = pd.Categorical(df['mo_sub'])
    df_dummies = pd.get_dummies(df['mo_sub'], prefix = 'c')
    df = pd.concat([df, df_dummies], axis=1)

    df['mo_str'] = pd.Categorical(df['mo_str'])
    df_dummies = pd.get_dummies(df['mo_str'], prefix = 'c')
    df = pd.concat([df, df_dummies], axis=1)
    
    df.drop(columns=['submit date','start date'],inplace=True)
    df['base salary']=df['base salary'].apply(lambda x: float(x.strip('$').replace(',','')))

    return(df)

def get_user_agents():
    software_names = [SoftwareName.CHROME.value,SoftwareName.FIREFOX.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
    user_agents = user_agent_rotator.get_user_agents()

    user_agent = user_agent_rotator.get_random_user_agent()
    headers = {'User-Agent': user_agent}
    return(headers)

def get_proxies():
    proxies_req = requests.get('https://www.sslproxies.org/')
    soup = bs.BeautifulSoup(proxies_req.text, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')

    proxies = [ el.find_all('td')[0].string+':'+el.find_all('td')[1].string for el in proxies_table.tbody.find_all('tr') ]
    proxy=choice(proxies)
    
    proxy_dict={"http": proxy, "https": proxy}
    return(proxy_dict)

def get_pages_no():
    lnk = "https://h1bgrader.com/"+params['company']+"/j/c/y"
    page = requests.get(lnk)
    soup = bs.BeautifulSoup(page.text, 'lxml')
  
    lnk_pages = soup.find_all('a', {'class':'page-link'})
    pages = [ el.get('href').split('page=') for el in lnk_pages ]
    nums = max([ int(el) for el in list(itertools.chain.from_iterable(pages)) if 'https' not in el ])
    return(nums)

def get_data():
    num_pages = get_pages_no()
    lnks = [ "https://h1bgrader.com/"+params['company']+"/j/c/y?page="+str(el) for el in range(num_pages) ]    
    hdrs = get_user_agents()

    if(params['distr']=='async'):
        responses = (grequests.get(el,headers=hdrs) for el in lnks)
        pages = grequests.map(responses)
    elif(params['distr']=='thread'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(load_url_thread, url, hdrs): url for url in lnks}

        responses = (future.result() for future in concurrent.futures.as_completed(future_to_url))
        pages = grequests.map(responses)
    elif(params['distr']=='sync'):
        pages = [ requests.get(el,headers=hdrs) for el in lnks ]
   
    docs = [ lh.fromstring(el.content) for el in pages ]
    elements = [ el.xpath('//*[@id="data"]') for el in docs ]
    
    columns = ['employer', 'job title', 'base salary', 'location', 'submit date', 'start date', 'case status']
                     
    df = pd.DataFrame()
    for el in elements:
        data = [ k.text_content() for k in el[0][1].iterchildren() ]
        raw_data = [ list(filter(None, re.sub(r'([\t]+|[ \n]+)', ' ', l).split('  '))) for l in data ]
        out = DataFrame.from_records(raw_data)
        df = pd.concat([df,out],axis=0,ignore_index=True)

    df.columns = columns
    return(df)

def load_url_thread(url,hdrs):
    return(grequests.get(url,headers=hdrs))
                
def grouping(df,index):
    grp = df.groupby([str(index)]).agg(\
                                      bs_sum=pd.NamedAgg(column='base salary', aggfunc='sum'),\
                                      bs_mean=pd.NamedAgg(column='base salary', aggfunc='mean'),\
                                      bs_max=pd.NamedAgg(column='base salary', aggfunc='max'),\
                                      job_cnt=pd.NamedAgg(column='job title', aggfunc='count'),\
    )

    grp.reset_index(inplace=True)
    return(grp)
    
def viz_data(df,xtitle,ytitle):
    df.plot(
        x=str(xtitle), 
        y=str(ytitle), 
        kind='bar', 
        legend=False, 
        color='blue',
        width=0.8,
        figsize=(16,9)
    )

    plt.xlabel(str(xtitle))
    plt.ylabel(str(ytitle))
    plt.gca().yaxis.grid(linestyle=':')
    plt.xticks(rotation='vertical', fontsize = 8)
    plt.savefig(params['company']+'.pdf')
    plt.show()
    
if __name__ == '__main__':
    if(params['scrape']):
        df_raw = get_data()
        write_to(df_raw,params['company'],'csv')
    else:
        df_raw = read_data(str(fl)+'.csv')
    
    check_missing_data(df_raw)

    df = process_cols(df_raw)
    grp = grouping(df,params['index'])
    viz_data(grp,params['index'],'bs_max')
