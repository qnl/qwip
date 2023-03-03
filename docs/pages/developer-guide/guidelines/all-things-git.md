# All Things Git

## Making a Pull Request

### General Procedure
In the following, we'll walk through how to correctly make changes to existing code by creating a new branch and pull request. To make this guide more clear, a running example will be used throughout. 

In our scenario, we want to update our package dependencies in the setup.cfg file of our repository. So we start by...

1. **Create a new branch** with a name that describes the changes you want to make. This new branch will contain your edits and won't affect the main branch unless it is approved later on. <br />
    - *Command:* `git checkout branch-name`<br />
    - Example: `git checkout git-docs`

2. **Check if on correct branch.** Make sure your changes aren't going on the main branch. <br />
    - *Command:* `git status`<br />

3. **Push branch to online repository.** This will create a new branch visible to everybody else in the repository. <br />
    - *Command:* `git push -u origin branch-name`<br />
    - Example: `git push -u origin git-docs`

4. **Make edits!** <br /> When writing code, follow good practices such as:
    - Include tests with ideally full coverage (See *pytest* section for more info)
    - Add detailed documentation (*mkdocs*)

5. **Add your changes.** Adding will not make any permanent changes, including the branch, and instead set them up later to be committed. <br />
    - *Command:* `git add` (add file name if want to be specific)<br />
    - Example: `git add setup.cfg`

6. **Commit changes.** Your changes are now a part of the branch you are currently in but are not visible on GitHub. <br />
    - *Command:* `git commit -m "Descriptive message of commit being made"`
    - Example: `git commit -m "Added missing dependencies to setup.cfg"`

7. **Update online repo with committed changes.** Now if you want people to be aware of the changes you have made in this new branch, you need to "push" it. *Reminder:* whenever you want something to be visible online (not local), we "push."
    - *Command:* `git push origin branch-name`
    - Example: `git push origin git-docs`

8. **Make pull request.** This is when you would like your changes to be a part of the main branch. 
    - *Command:* Go on github and click on the option to create a pull request and include a description of your edits. Optionally assign someone to evaluate your changes. 

9. **Sync your local system with the shared repository.**  Ensures that you have the most updated information. 
    - *Command:* `git pull` (pulls everything down from repo)
    - Alternative Command: `git fetch` (tells you what changes have been made without applying them)


### Cloning a Repository
Follow these steps to link your local machine with the repository.

1. Access the desired repository on GitHub and select the option clone using an SSH key and copy the link. 

2. In the command prompt, go to the desired folder you would like the repo to be in using `cd`. 

3. To clone the repository in the folder you're currently in:
    - *Command:* `git clone ssh-key` (Paste the specific key in replace of `ssh-key`)

4. *(Optional -- Installing Dependencies)* If your repo has a setup.cfg file, run:
    - *Command:* `pip install -e .` (all) or `pip install -e .[tests,dev]` (specific)

### Creating an SSH Key
Reference: [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)