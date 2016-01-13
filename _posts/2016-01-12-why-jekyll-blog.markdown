---
layout: post
title:  "Why Jekyll for Blogging?"
date:   2016-01-12 19:52:10 +0700
categories: jekyll update
---
Several factors that I weighed in as I chose my toolkit for blogging: 

1. **Simplicity** - Whatever tool I end up using has to be drop dead easy. 
2. **Cost** - I know shared hosting is affordable, but for an experimentation project, I would rather go cheap. 
3. **Language** - I prefer Javascript and Python, since those are the two languages I have been learning to program in. 
4. **Design** - This is a wild card but I need to like the look of my blog!

Now let's go through them one by one. 

*Online blogs -  (blogger, tumblr, wordpress.com, etc):* These guys are free and often dropdead simple. They are meant for bloggers who do not want to get into technical things. My issue with them is about fixed design. Blogger has no serious themes to choose from. WordPress.com had limited free themes that are very difficult to customize. Tumblr felt like Instagram to me and I dislike Instagram. Medium is more interesting and its distribution system made it very appealing. It didn't quite suit my purpose this time though as you can only write on Medium whereas I wanted a multi-functional personal site with an emphasis on blogging.

*CMS software - (WordPress software):* This is an interesting option. WordPress requires that you find your own hosting, which is not preferable. Functionally, WordPress can do A LOT of things and indeed ["over 23.3% of the top 10 million websites"](wp-stat) in 2015 were built with WordPress. For almost anything I wanted to do, I could find a plugin built by the WordPress community. The downsides are that maintenance is harder and WordPress sites also seem slow even for simple ones like my cousin's personal-sound engineer portfolio site, **[Teera Music](teera)**. For my site, I just want text with a few image and would rather not deal with the whole package.

*Static site generator - (Jekyll, Hyde, Assemble, Pelican):* Static site generators turns out sit in sweet spot between online blogs and cms softwares, offering the right trade-off between customizability and simplicity. Jekyll, in particular, is the grandfather of static site generators made for the purpose of blogging. It was built by the founder of GitHub and works seamlessly with GitHub pages, making free hosting via GitHub possible. Once you have it set up, you can just write your blog in Markdown (like LaTex) along with a few lines of command. If you are not afraid of using command lines (`jekyll serve`) and navigating through folders that start with underscroll (`_posts`), I would argue that static site generators is way easier to deal with than the WordPress cms. 

Lastly, Jekyll also offers powerful support for code snippets:

{% highlight python %}
def printHi (name):
    print "Hi " + name
    return

printHi(Pete)
#### prints 'Hi Pete' to the STDOUT
{% endhighlight %}

If you're interested do check out [Jekyll](jekyll). 

[jekyll]: http://jekyllrb.com
[teera]: http://www.teeramusic.com/