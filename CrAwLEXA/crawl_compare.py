#!/usr/bin/env python

import json
import datetime
import tldextract
import calendar
import csv
import codecs 


def extract(x):
    try:
        return str(tldextract.extract(x).registered_domain)
    except:
        return "#totallyUniqueStringJusttestingHere#"

def extract_fqdn(x):
    try:
	do = tldextract.extract(x)
	if do.registered_domain:
	    if do.subdomain:
		return str(do.subdomain + '.' + do.registered_domain)
	    else:
		return str(do.registered_domain)
    except: 
	return "#totallyUniqueStringJusttestingHere#"

def main():
    data_initial = open("new-firstday-blacklist", "rU")
    reader_blacklist = csv.reader((line.replace('\0','') for line in data_initial), delimiter=",")
    blacklist = {}
    for row in reader_blacklist:
        key = extract_fqdn(row[0])
	if int(row[1]) > 1457568000:
            if key in blacklist:
                if row[1] < blacklist[key]:
                    blacklist[key] = row[1]
            else:
                blacklist[key] = row[1]
    print len(blacklist)
    data_init = open("urls_in_page.tsv", "rU")
    reader_active = csv.reader((line.replace('\0','') for line in data_init), delimiter=" ")
    active = {}
    for row in reader_active:
	#print len(row)
	if len(row) > 1:
            key = extract_fqdn(row[0])
	    #print key
            if key in blacklist:
                if key in active:
		    #if row[1] < active[key]:
                    active[key] = row[1]
                else:
                    active[key] = row[1]
    print len(active)
    time_diff = {}
    comp_output = []
    for k, v in active.iteritems():
        #temp = int(active[k]) - int(blacklist[k])
	time_diff[k] = active[k]
        # print time_diff[k]
	#comp_output.append([str(k),  str(temp), str(active[k]), str(blacklist[k])])
    with open('crawl-resources-bl-fqdn.tsv', 'w') as fp:
    #    json.dump(comp_output, fp, sort_keys=True, indent=4)
	for k, v in time_diff.iteritems():
	    fp.write(str(k) + '\t' + str(time_diff[k]) + '\t' + str(active[k]) + '\t' + str(blacklist[k] + '\n'))


if __name__ == '__main__':
    main()

