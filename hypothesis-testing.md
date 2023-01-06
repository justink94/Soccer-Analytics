## This hypothesis test will test the theory that shooting more often regardless of position will help yield more wins for a team. In other words, don't try and get into perfect position for every shot. Take as many as you can and you are bound to catch a defender or goal keeper off guard at some point.

```{r}
library(tidyverse)
```

### Create 'all teams' frame by binding together all 5 hypothesis test frames you made with the python hypo test function previously

```{r}
all_teams <- rbind(all_teams, spain_hypo)
```

### $H_0$ = Teams that shoot higher than average do not win more than average $\mu = 14.07$
### $H_1$ = Teams that shoot higher than average do win more than average $\mu > 14.07$
### $\alpha$ = 0.05

### We have a data frame of all teams that shot above average and their amounts of wins across Europe. The mean wins for all teams is 14.07. I am going to test to see whether teams that shot more have significantly more wins that teams that did not.

```{r}
shot_above = all_teams %>%
  filter(all_teams$`above average` == '1')
shot_below = all_teams %>%
  filter(all_teams$`above average` == '0')
```

### Even though our sample size is larger than 30, I am still going to use the Shapiro-Wilk normality test for the data. As we can see with an $\alpha$ = 0.05 and a p-value of 0.6649 we do not reject the null hypothesis of the data being normally distributed. So we can move forward with our test

```{r}
shapiro.test(shot_above$Wins)
```

### With an $\alpha$ of 0.05 and a p-value of 4.392e-05, we reject the null hypothesis of teams that shoot more than average to not win more than average. 

```{r}
t.test(shot_above$Wins, mu = 14.07, type = 'one tailed')
```

