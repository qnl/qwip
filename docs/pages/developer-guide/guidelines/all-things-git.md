# All Things Git

Git is a version control system used for tracking changes to code and collaborating with others on development. While there is no need to be an expert on git, it certainly helps to have a working understanding. There are plenty of good references to git on the internet already, which we link to below, rather than duplicating.

<figure markdown>
  <a href="https://xkcd.com/1597/"><img src="https://imgs.xkcd.com/comics/git.png"></a>
  <figcaption>If that doesn't fix it, git.txt contains the phone number of a friend of mine who understands git. Just wait through a few minutes of 'It's really pretty simple, just think of branches as...' and eventually you'll learn the commands that will fix everything.</figcaption>
</figure>

## References

[Git](https://git-scm.com/doc) - The official git documentation

[Atlassian](https://www.atlassian.com/git) - Atlassian has a comprehensive series of tutorials and introductions to using git and version control.

[git-sim](https://initialcommit.com/blog/git-sim) - Git-Sim is a python package for visualizing branches, commit graphs, and what each git command does to the commit graph.

[Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

## Best Practices

1. *Keep commits small and atomic. Commit often.*

    Committing often and keeping changes small allows you to easily revert a specific change or go back to a particular state of the repository.

2. *Write useful commit messages.*

    Commit messages should describe what was changed. "Bug fix" is much less clear than "Fixed off by one error in sequence indexing`

4. *Run code in `main`.*

    When everyone works by default in a separate branch, changes tend to accumulate and branches increasingly diverge. Eventually you reach a point where every member of the team is working with a different version of the library. Instead make branches only when contributing a specific feature or fix to the library.

3. *Make changes in separate branches.*

    Changes should not be made directly in the main branch, but in feature branches or bug fix branches. For new features, branches should be named `feature/name-of-feature`. For bug fixes or patches, branches should be named `fix/description-of-fix`

4. *Submit pull requests for all changes.*

    Pull requests allow people to review code before it is merged, and provides a forum for discussing proposed changes.


## Making a Pull Request

Merging to the `main` branch should generally be done through a pull request. This allows changes to be reviewed and discussed before being added to the library. This also allows for automated testing and code checks to be run on the repository.

In the following, we'll walk through how to correctly make changes to existing code by creating a new branch and pull request. To make this guide more clear, a running example will be used throughout. 

In our scenario, we want to update some documentation in this file of our repository.

<div class="termy">
```console
$ git checkout -b docs/git
Switched to a new branch 'docs/git'
$ git status
On branch docs/git
nothing to commit, working tree clean
$ # After making some edits...
$ git status
On branch docs/git
Changes not staged for commit:
    (use "git add <file>..." to update what will be committed)
    (use "git restore <file>..." to discard changes in working directory)
        modified:   docs/pages/developer-guide/guidelines/all-things-git.md
no changes added to commit (use "git add" and/or "git commit -a")
$ git add -A
$ git commit -m "Added pull request tutorial in git documentation"
[docs/git 061f169] Added pull request tutorial in git documentation
1 files changed, 200 insertions (+) 20 deletions(-)
$ git push -u origin docs/git
---> 100%
```
</div>

1. **Create a new branch** with a name that describes the changes you want to make. This new branch will contain your edits and won't affect the main branch unless it is approved later on. The `-b` flag creates a new branch in addition to checking out a branch. If you've already made some changes to the code you can still checkout a *new* branch and these changes will stay.

    *Command:* `git checkout -b [branch-name]`

2. **Check if on correct branch.** Make sure your changes aren't going on the main branch.
    
    *Command:* `git status`

3. **Make edits!** Make sure you follow the coding, testing and documentation guidelines outlined in the subsequent sections.
    
4. **Add your changes.** Adding will *stage* the changes and set them up to be subsequently commited. If you want to split your changes into several commits, you can add a subset of files and then commit those changes first before moving on to the other files.
    
    *Command:* `git add [filename]`

6. **Commit changes.** Your changes have been added to the current branch but have not been synced with the remote repository on GitHub.

    *Command:* `git commit -m "Useful commit message."`

7. **Update online repo with committed changes.** Update the remote repository on GitHub, you need to "push" it. If this branch has just been created and does not yet exist on the remote repository, use the `-u` flag to create a matching branch in the remote repository. *Reminder:* whenever you want something to be visible online (not local), we "push."

    *Command:* `git push -u origin [branch-name]` or `git push`

8. **Make pull request.** Open a pull request whenever you are ready for changes to be discussed. If you are not yet ready for the changes to be merged you can also make a "draft" pull request. After someone review the code and all changes are approved, you can merge the changes back to the main branch. Choose the "Squash and Merge option".

9. **Sync your local system with the remote repository.**  This ensures that your local repository is up to date with the remote. If you just want to check what changes exist on the remote repository, you can run `git fetch` to get updates without applying them.
    
    *Command:* `git pull`


## Cloning a Repository
Follow these steps to link your local machine with the repository.

1. Access the desired repository on GitHub and select the option clone using ssh (not https) and copy the link. 

2. In the command prompt, go to the desired folder you would like the repo to be in using `cd`. 

3. To clone the repository in the folder you're currently in:

    *Command:* `git clone [link-to-repository].git` 

