#!/bin/bash

while :
do
    scrapy crawl alexa_top_1m >> logs/crawl.log &
    strace -o trace/$(date +%s.%N) -p $! -f -e trace=network >> logs/strace.log &
    sleep 2h
    python urlDiff.py >> logs/diff.log &
    sleep 2h
done
