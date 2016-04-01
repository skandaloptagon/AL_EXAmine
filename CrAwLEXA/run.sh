#!/bin/bash

while :
do
    scrapy crawl alexa_top_1m -a n=1000 &
    strace -o trace/$(date +%s.%N) -p $! -f -e trace=network >> logs/strace.log &
    sleep 1h
done
