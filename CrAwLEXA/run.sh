#!/bin/bash

while :
do
    scrapy crawl alexa_top_1m &
    strace -o trace/$(date +%s.%N) -p $! -f -e trace=network
    sleep 2h
    python urlDiff.py &
    sleep 2h
done
