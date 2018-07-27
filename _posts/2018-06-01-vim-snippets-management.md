---
layout: post
title: "Vim Snippets Management with Ultisnips"
author: "Sangsu Lee"
categories: journal
tags: [documentation,sample]
image: vimlogo.jpg
---

Welcome, Vim lovers!  
[Ultisnips](https://github.com/SirVer/ultisnips)
by SirVer, currently stands one of the most popular vim plugin.
It provides concise, comprehensive snippet syntax which allows vim users to
easily customize their snippets, and you can also use python interpolation on snippet syntax.  
But in this post, rather than introducing nice features of Ultisnips itself,  
I'll share how I manage my vim snippets efficiently in everyday workflow, with Ultisnips.

## Snippets?

<img src="{{ site.github.url }}/assets/img/gif_ultisnips.gif">
_Deoplete.nvim + snippet integration_

Snippet expanding is one of the popular features that most of the other IDEs have as a default.
Probably you've tried to triggered them by hitting \<tab\>, or you mapped other key binding
for snippets to span in many other IDEs.  
Obviously this is something that we can't live without when we write code(At least for me).

## Install Ultisnips, and default vim-snippets

I assume we all know how to install plugin either by downloading source file,
or using plugin manager. If you're not, I recommend you to give
[this plugin manager](https://github.com/junegunn/vim-plug) a try.
Personally I've used [Vundle](https://github.com/VundleVim/Vundle.vim),
but recently have switched it to [vim-plug](https://github.com/junegunn/vim-plug) because of
its simplicity, and many nice features of it. Anyway,
Here are github links for two plugins we're going to use today.

-   [SirVer/ultisnips](https://github.com/SirVer/ultisnips)
-   [honza/vim-snippets](https://github.com/honza/vim-snippets)

To install, put these lines in your ~/.vimrc and execute :PlugInstall

```viml
" If you prefer to use Vim-plug:
Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets'
" And inside of vim editor, execute :PlugInstall
```

Reason why we install both two is, because Ultisnips itself does not provide
off-the-shelf default snippets for every language, but we need to define them from the scratch.  
That's why we need to have [honza/vim-snippets](https://github.com/honza/vim-snippets) which provides default
boilerplate snippets of many languages. Repository has separate snippet directories, and
each is compatible with Ultisnips, vim-snipmates, or other snippet plugins. We're going to tweak,
or append our own snippets on top of existing ones.  
Now that we have our plugins installed on your vim, we're good to go.
default trigger for Ultisnips is, \<tab\>. try some snippets out,
and if you want autocomplete features also, you can take a look at
[Deoplete](https://github.com/Shougo/deoplete.nvim), or
[YCM](https://github.com/Valloric/YouCompleteMe) integration.

## Managing snippets in your own snippets repository

Setting development environment is tedious, often time-consuming. I think we can all agree with this.
Every time we change our environment like laptop to desktop, or ssh into other machine, we synchronize them
so that our different environments work more consistently.
But building such workflows definitely annoys us in many ways.  
That's why we have our own repository named dotfiles for storing
shell config file, .vimrc file, .tmux file, etc. And no exception for snippets.  
I think we'd better to have our own repository for snippets.  
Steps are following:

1.  Remove existing whole vim-snippets directory in your ~/.vim/bundle
2.  Fork vim-snippets repository (you can notice there are many people like us, judging from the fork count)
3.  Instead of putting original repository of vim-snippets, put yours.

In shell :

```bash
rm -rf ~/.vim/bundle/vim-snippets
# Or, you can remove the line "Plug 'honza/vim-snippets'" in .vimrc,
# and execute :PlugClean in vim.
```

In .vimrc :

```viml
" After fork the repository,
Plug 'SirVer/ultisnips'
Plug '<your_github_id>/vim-snippets'
" And inside of vim editor, execute :PlugInstall again.
```

Now we have our own snippet repository, ready to be modified, or be deployed anywhere.
Keep your repository updated with your own snippets, and to pull the changes from your snippet repository to other machine, just execute :PlugUpdate.

## Separate vim-snippets and your private snippets

Now you can directly modify, or append extra snippets to existing vim-snippets snippet file. Now it's yours.  
But if you want to separate the snippets in vim-snippets with user-specific snippets
(which is like, snippets that could be used for algorithm competition), make empty folder at
your dotfiles directory and store your snippets in it.  
And to make the plugin recognize those snippets, you should let vim and Ultisnips know where our private snippets are placed.  
This is done by adding path to 'runtimepath' value, and declare some global variables for Ultisnips.  
You can find more detailed explanation in Ultisnips docs, ':h Ultisnips'

```viml
" Our personal snippets go into ~/dotfiles/user_snippets.
" By defining this, ':UltiSnipsEdit' call opens new file at this location
let g:UltiSnipsSnippetsDir="~/dotfiles/user_snippets"

" Add your private snippet path to runtimepath
set runtimepath^=~/dotfiles
" When vim starts, Ultisnips tries to find snippet directories defined below, under the paths in runtimepath.
let g:UltiSnipsSnippetDirectories=["UltiSnips", "user_snippets"]
```

## Small Tip

By binding :UltiSnipsEdit command to other keymap, you can modify your private snippets
more easily, and quickly.

```viml
nnoremap <leader>es :UltiSnipsEdit<cr>
```

My leader key is bound to \<space\>, so \<space\>-e-s opens private snippet file for the
current filetype on same terminal, in splitted vim window.
