import React from 'react';
import Container from '@mui/material/Container';
import useMediaQuery from '@mui/material/useMediaQuery';
import { useTheme } from '@mui/material/styles';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Divider from '@mui/material/Divider';
import { useTranslation } from 'next-i18next';
import logo from '/public/images/profile-logo.svg';
import brand from '/public/text/brand';
import { useText } from '/theme/common';
import useStyles from './footer-style';
import ContactForm from '../Contact/Form';

function Footer() {
  const { classes } = useStyles();
  const { classes: text } = useText();
  const { t } = useTranslation('common');

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <div className={classes.root}>
      <Container maxWidth="lg" component="footer">
        <Grid container spacing={6} direction={isMobile ? 'column-reverse' : 'row'}>
          <Grid item xs={12} md={5}>
            <div className={classes.logo}>
              <img src={logo} alt="logo" />
              <Typography variant="h3" className={text.title}>
                {brand.profile.name}
              </Typography>
              <Typography variant="h4" className={text.subtitle}>
                {brand.profile.title}
              </Typography>
            </div>
            <div className={classes.socmed}>
              <IconButton aria-label="Delete" className={classes.margin} size="small">
                <i className="ion-logo-facebook" />
              </IconButton>
              <IconButton aria-label="Delete" className={classes.margin} size="small">
                <i className="ion-logo-instagram" />
              </IconButton>
              <IconButton aria-label="Delete" className={classes.margin} size="small">
                <i className="ion-logo-twitter" />
              </IconButton>
              <IconButton aria-label="Delete" className={classes.margin} size="small">
                <i className="ion-logo-linkedin" />
              </IconButton>
            </div>
            <div className={classes.contact}>
              <Typography className={text.paragraph}>
                {t('profile-landing.footer_contact')}
                <br />
                +12 345 678 90
              </Typography>
              <Divider className={classes.divider} />
              <Typography className={text.paragraph}>
                {t('profile-landing.footer_hello')}
                <br />
                jenadoe.skype
              </Typography>
            </div>
          </Grid>
          <Grid item xs={12} md={7}>
            <ContactForm />
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default Footer;
