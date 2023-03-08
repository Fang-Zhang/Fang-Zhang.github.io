import React, { Fragment } from "react";
import PropTypes from "prop-types";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useTheme } from "@mui/material/styles";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import { useTranslation } from "next-i18next";
import SideNavigation from "../SideNavigation";
import brand from "/public/text/brand";
import { useText } from "/theme/common";
import Settings from "../Settings";
import useStyles from "./banner-style";

function BannerNav(props) {
  const theme = useTheme();
  const { classes: text } = useText();
  const { classes } = useStyles();
  const isDesktop = useMediaQuery(theme.breakpoints.up("lg"));
  const isTablet = useMediaQuery(theme.breakpoints.down("lg"));
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const { t } = useTranslation("common");
  const { onToggleDark, onToggleDir } = props;

  return (
    <div className={classes.root}>
      <Container fixed={isDesktop}>
        <Grid container spacing={0}>
          <Grid item lg={2} xs={12}>
            {!isTablet && <SideNavigation />}
          </Grid>
          <Grid item lg={10} xs={12}>
            <div className={classes.banner}>
              <div className={classes.cover}>
                <div className={classes.figure}>
                  {/* <img src={brand.profile.cover} alt="cover" /> */}
                  <div className={classes.overlay} />
                </div>
              </div>
              {/* {!isTablet && (
                <div className={classes.settingIcon}>
                  <Settings toggleDark={onToggleDark} toggleDir={onToggleDir} />
                </div>
              )} */}
              <div className={classes.text}>
                <Typography variant="h4" className={text.title2}>
                  {t("profile-landing.banner_greeting")}
                </Typography>
                <Typography variant="h2" className={text.title}>
                  {t("profile-landing.banner_me")}
                </Typography>
                {!isMobile && (
                  <Fragment>
                    <Typography variant="h5" className={text.subtitle2}>
                      {t("profile-landing.banner_desc")}
                    </Typography>
                    <div className={classes.socmed}>
                      <IconButton
                        aria-label="Delete"
                        className={classes.margin}
                        size="small"
                        href="https://github.com/Fang-Zhang"
                        target="_blank"
                        style={{ marginRight: 1 + "em" }}
                      >
                        <i className="ion-logo-github" />
                      </IconButton>
                      <IconButton
                        aria-label="Delete"
                        className={classes.margin}
                        size="small"
                        href="https://twitter.com/WalterFangZhang"
                        target="_blank"
                        style={{ marginRight: 1 + "em" }}
                      >
                        <i className="ion-logo-twitter" />
                      </IconButton>
                      <IconButton
                        aria-label="Delete"
                        className={classes.margin}
                        size="small"
                        href="https://www.linkedin.com/in/fang-zhang-ba76aaa4/"
                        target="_blank"
                        style={{ marginRight: 1 + "em" }}
                      >
                        <i className="ion-logo-linkedin" />
                      </IconButton>
                      {/* <IconButton aria-label="Delete" className={classes.margin} size="small" href="mailto:walter.zhangfang@gmail.com" target="_blank">
                        <i className="ion-logo-google" />
                      </IconButton> */}
                      {/* <IconButton aria-label="Delete" className={classes.margin} size="small" href="mailto:walter.zhangfang@gmail.com" target="_blank">
                        <i className="ion-logo-facebook" />
                      </IconButton> */}
                    </div>
                  </Fragment>
                )}
              </div>
            </div>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

BannerNav.propTypes = {
  onToggleDark: PropTypes.func.isRequired,
  onToggleDir: PropTypes.func.isRequired,
};

export default BannerNav;
