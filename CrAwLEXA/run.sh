#!/bin/bash

while :
do
    scrapy crawl alexa_top_1m -a n=100000 &
    strace -o trace/$(date +%s.%N) -p $! -f -e trace=network >> logs/strace.log &
    sleep 2h
done
