#!/bin/bash
sleep 5m

while :
do
    scrapy crawl alexa_top_c &
    strace -o trace/$(date +%s.%N) -p $! -f -e trace=network >> logs/strace.log &
    sleep 4h 
done
