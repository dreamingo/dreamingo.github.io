#!/bin/bash

read -r -d '' errorMsgParam <<- EOM
new_post.sh: Missing the post-name
[Usage]: bash new_post.sh post-name
EOM

if [ ! $# -eq 1 ];then
    echo "$errorMsgParam"
    exit -1
fi

postName=$1
fileName=$(date +%F)-${postName}.md

if [ -f _posts/${fileName} ];then
    echo "File _post/${fileName} has been existed!"
    exit -1
fi


cat > _posts/${fileName} <<EOL
---
layout: post
title: "${postName}"
modified:
categories: 
description:
tags: []
image:
    feature: abstract-10.jpg
    credit:
    creditlink:
comments: true
mathjax: false
share:
---
EOL
