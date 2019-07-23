---
  title: Mapping the Underlying Social Structure of Reddit
author: Andrew Carr
date: '2019-07-22'
slug: mapping-the-underlying-social-structure-of-reddit
categories: []
tags: []
---
  
  Since it was founded in 2005, Reddit has developed into a popular place for sharing opinions and ideas, rating web content, and aggregating news on the internet.  Reddit is organized into thousands of user-made communities, called subreddits, which cover a broad range of subjects, including politics, sports, technology, personal hobbies, and self-improvement, to name a few. Given that Reddit is structured in this way, it is natural to think of Reddit as a population of users organized into many overlapping communities (subreddits).  In other words, it makes sense to conceptualize Reddit as having an underlying social structure.  Uncovering this structure may provide insights into the social organization of internet culture more generally.  

My goal in this post is to map the social structure of Reddit by measuring the proximity of Reddit communities to each other.  I'm operationalizing proximity between subreddits as the number of submissions on these subreddits that come from the same user in a given time period.  For example, if a user posts something to subreddit A and then a few days later posts something else to subreddit B, subreddits A and B are linked by this user.  Subreddits with more users in common are closer together.  The idea that networks between groups are formed by the people these groups have in common is an old one in sociology (Breiger 1974); more recently, it has served as a conceptual basis for producing networks from internet data (https://www-cs.stanford.edu/~jure/pubs/cesna-icdm13.pdf, https://cs.stanford.edu/people/jure/pubs/agmfit-icdm12.pdf).  To my knowledge, this idea hasn't been used to examine the social structure of communities on Reddit, although other methods have been used to analyze the community structure of Reddit (Olson and Neal 2015).  

# Data 

The data I'm going to use for this post come from a marvelous online repository of subreddit submissions, comments, and other content that is generously hosted by data scientist Jason Baumgartner.  Although I'm only making use of the submissions section of Baumgartner's repository, I've downloaded a lot of his data.  If you plan to download a lot of data from this repo, I implore you to donate a bit of money to keep Baumgartner's database up and running (donate here - https://pushshift.io/donations/).  Hosting this data is not free!    

Here's the link to Baumgarter's Reddit submissions data - http://files.pushshift.io/reddit/submissions/.  Each of these files has all Reddit submissions for a given month between June 2005 and May 2019.  Files are JSON objects stored in various compression formats that range between .017Mb and 5.77Gb in size, a testament to how much Reddit has grown over the years.  Let's get our feet wet with this data by downloading something in the middle of this range - the 710Mb file for all Reddit submissions in March 2013.  This file is called RS_2013-05.bz2. It will take a minute or to download.  You can double-click this file to unzip it, or if you prefer to work in the Terminal use the following command - bzip2 -d RS_2013-05.bz2.  This will take a couple of minutes to unzip.  Make sure you have enough room to store the unzipped file on your computer - it's 4.51Gb.  

Once we have unzipped this file, we can use the read_lines function from the readr function to preview the data.

```{r}
library(readr)
read_lines("RS_2013-05", n_max = 10)
```
