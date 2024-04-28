---
layout: post
cover:  assets/images/declutter_websites/before_and_after_youtube.jpg
title:  Give social media websites a productive redesign
categories: [ blog, productivity ]
comments: true
---

<!-- What websites do you spend considerable time on? I'm going to throw out 3 names, and I'm betting I get at least one of them right: Facebook, Netflix, Tiktok. I'm cheating of course -->

<br>

## A look at the numbers

We spend **a lot** of time online. According to a [HootSuite reoport](report), that number is about 6 hours 42 minutes per day. Are you ready for the real kicker? These are **2019, prepandemic numbers**. That means we spend an average of 100 days online every year.

Do we spend so much time online because we live in an increasingly digital world<!-- which requires computer access for our work and communication -->, or is it because *well, it's fun and maybe slightly addictive*? The data indicates both (though we can already infer this by analyzing our own habits).

A [Nielsen Report](https://www.nielsen.com/wp-content/uploads/sites/3/2019/04/nielsen-social-media-report.pdf) found that Americans spend 23% of their time on social media and blogs, which means we spend 23 days online every year browsing Facebook, YouTube, and Instagram. Note that the report explicitly categorized *instant messaging* (WhatsApp, Telegram, Discord) as a separate category (at 3.3% of online time). This means that the majority of the time spent on social media is unlikely to be just communicating with other people. Most likely other activities, such as browsing, represent a significant fraction of that time.

How can we minimize time online, and still reap most of the benefits?

<br>

## How to augment your digital experience

Online businesses have a natural incentive to keep you on their websites for as long as possible. The majority of the Internet, notably all major social media platforms, use advertisements to generate revenue. This gives them an obvious incentive to keep you engaged: the more you stay on the website, the more advertisements you can click on. Even the websites that don't use an ad-driven model often have the same incentive. Netflix wants you to spend more time on the platform so that you perceive the value it brings and keep buying a subscription. The more Amazon keeps you on the platform, the more products they can recommend you.

As a result, the tech industry has developed powerful and often simple design patterns that are incredibly effective at maximizing engagement: infinite scrolling, autoplay, recommendation lists, notification bubbles, alerts, and others.

In this blog I will show you how we can decrease the effectiveness of these patterns using some free and [open-source](https://en.wikipedia.org/wiki/Open_source) browser extensions.

The two browser add-ons I will be using are available for Firefox, Chrome and Safari:

* [**LeechBlock**](https://www.proginosko.com/leechblock/): *LeechBlock* NG is a free productivity tool which allows you to manage time-wasting sites: You can block websites, limit the amount of time you spend on them, automatically redirect yourself to another website, and many other nifty features. I will be using it to redirect webpages. *Note*: I used to browse reddit whenever I was bored. At one point I distinctly remember opening the browser for a google search and catching myself typing the reddit address out of habit. I kicked the habit by using *LeechBlock* to redirect any requests to reddit directly to my university course page every time I wanted to procrastinate.
* [**uBlock Origin**](https://ublockorigin.com/): *uBlock Origin* is primarily used as an "ad blocker". Ad blocking is its own can of worms so I will not go into it, but I want to show one of *uBlock*'s lesser known tools: deleting content on a webpage.

<br>

### Useful Redirects

We can use *LeechBlock*'s redirection feature to automatically switch to a less distracting version of the websites we're visiting (if one exists).

#### Facebook.com

Use *LeechBlock* to redirect *facebook.com* to *messenger.com*. If your main use for Facebook is messaging your friends, *messenger.com* is Facebook's option that gives you **just** that.

![Redirecting Facebook to Messenger](/assets/gifs/declutter_websites/facebook_to_messenger.gif)

#### LinkedIn.com

Use *LeechBlock* to redirect *linkedin.com* to *linkedin.com/messaging*, which achieves the same purpose as *messenger.com* for Facebook.

#### Reddit.com

Change *reddit.com* for *old.reddit.com*. This takes you back to Reddit's old design, which has pages instead of an infinite newsfeed. It doesn't use many other modern sticky techniques, presumably because it was just easier to implement most of them on the newer design. Unfortunately, I don't think this is doable with *LeechBlock* so you would have to just be mindful of browsing reddit from *old.reddit.com*.

![Reddit new and old layouts](/assets/images/declutter_websites/reddit_new_old.jpg)

#### How to set up *LeechBlock* to redirect webpages

1. Download *LeechBlock* on your browser from your browser extensions store.
2. Right click on the *LeechBlock* icon, and go into options.
3. Create one block per website as shown below.

<!-- <img src="/assets/images/declutter_websites/merged_leechblock.jpg" alt="Facebook and LinkedIn on *LeechBlock*" height="400px" width="auto"/> -->

![Facebook and LinkedIn on *LeechBlock*](/assets/images/declutter_websites/merged_leechblock.jpg)

Alternatively you can download [this file with configurations already set](/assets/extra/LeechBlockOptions.txt), and then import it into the extension by following the steps below.

![Importing settings file into *LeechBlock*](/assets/images/declutter_websites/annotated_leechblock.jpg){: width="350" }
<!-- <img src="/assets/images/declutter_websites/annotated_leechblock.jpg" alt="Importing settings file into *LeechBlock*" width="300"/> -->

<br>

### Redesigning webpages

In most cases, we will not have a webpage which we can easily redirect to for a less distracting experience. Instead, we can clean up the pages ourselves using *uBlock*'s Element Picker.

#### YouTube.com

We can use *uBlock* to delete elements from the page that make it easy to go down the YouTube rabbit hole. For me these are: *related videos* on the right side of the screen, the comment section, and the search bar. Therefore, let's delete them!

Below is an example of how we can use *uBlock* to stop the comment section from showing up.

![Deleting comments with *uBlock*](/assets/gifs/declutter_websites/deleting_comments_with_ublock.gif)

**Final Result**

These are the results after deleting all the elements I outlined above.

This is the view in the standard mode:
![View in standard mode](/assets/images/declutter_websites/after_ublock_in_standard_mode.jpg)

As you can see the only item left on the page is the video. We can further improve this by entering theater mode (by pressing `t` on your keyboard or by clicking on the theater icon on the bottom right corner of the video), which will enlarge the video to the majority of the screen.

![View in theater mode](/assets/images/declutter_websites/after_ublock_in_theater_mode.jpg)

Every time you use *uBlock* to delete an element from a page, it gets added to "My Filters" Tab in *uBlock*'s Options. Therefore, instead of manually deleting each of the elements you don't like, you can start by pasting my configurations directly into *uBlock*'s settings. Add the text below in the "My Filters" Tab in *uBlock*'s Options.

```
www.youtube.com###center
www.youtube.com###sections
www.youtube.com###start
www.youtube.com###end
www.youtube.com###menu-container
www.youtube.com###subscribe-button
www.youtube.com##.ytp-endscreen-content
www.youtube.com###meta-contents
www.youtube.com###masthead > .ytd-masthead.style-scope
www.youtube.com##ytd-watch-next-secondary-results-renderer.ytd-watch-flexy.style-scope > .ytd-watch-next-secondary-results-renderer.style-scope

```

Only include the code below if you want to also disable autoplay.

```
/watch_autoplayrenderer.js$script,domain=www.youtube.com
/annotations_module.js$script,domain=www.youtube.com
/endscreen.js$script,domain=www.youtube.com
```

![uBlock with youtube configurations](/assets/images/declutter_websites/ublock_with_youtube_changes.jpg)

Note that sometimes we will want to see the search bar, or perhaps the comments. In that case, we can click on the *uBlock* icon, and click on the big blue *power off* button which will disable it (you will have to re-enable it after manually).

![Disabling *uBlock*](/assets/images/declutter_websites/disabling_ublock.jpg)

<br>

## Conclusion

If you followed all of the above steps, you should now be redirected to *messenger.com* every time you browse *facebook.com*, you should see your messages on *linkedin.com/messaging* instead of the newsfeed from the default LinkedIn page, and your default *YouTube* experience is a lot cleaner. These are some of the augmentations you can do with *LeechBlock* and *uBlock* in order to declutter the UI of distracting websites whilst minimizing drawbacks. You can of course do so much more with them depending on how far you want to go. Take a look at all of *LeechBlock*'s settings to view the breadth of options at your fingertips, and any time you find yourself consistently distracted by some UI elements zap them with *uBlock*.

### More tips

* One of the most troubling design patterns is *phone notifications*. We check our phones at least [96 times a day](https://www.asurion.com/about/press-releases/americans-check-their-phones-96-times-a-day/), or once every 10 minutes. It is very easy to lose focus by checking any of the myriad of notifications we see on our screens, especially given that they are very rarely urgent or important. Try disabling **all but the absolutely necessary** notifications. Both Android and iOS allow you to modify notifications on a per-app basis.
* One cool feature of *LeechBlock* is to *delay* access to websites. This gives you a couple of extra seconds to reconsider whenever your muscle memory automatically takes you to one of your time-sinks. As an extra benefit, it also makes it more frustrating to open new content.
* Both extensions work on **mobile** Firefox.
